from torch import nn
import torch
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    Initializes a sparse MoE'
    layer according
    to the G-sharding paper
    """
    def __init__(self, d_model, n_experts, capacity_factor=1):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.ffn_dim = d_model*4
        self.capacity_factor = capacity_factor
        self.gates = nn.Linear(self.d_model, self.n_experts, bias=False)
    

        # Experts: We use nn.Parameter to manage the 3D weights directly (Experts, In, Out)
        # Layer 1: (Experts, D_model, FFN_dim)
        self.w1 = nn.Parameter(torch.zeros(n_experts, d_model, self.ffn_dim))
        # Layer 2: (Experts, FFN_dim, D_model)
        self.w2 = nn.Parameter(torch.zeros(n_experts, self.ffn_dim, d_model))
        
        # Initialization
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def forward(self, x):
        # 1. Route X to the router and softmax it
        # The dimension is the expert.
        # 1. Save original shape for later
        batch, seq, d_model = x.shape
        
        # Flatten the batch for routing: (Batch*Seq, d_model)
        x_flat = x.view(-1, d_model)

        # 2. Router (The "Brain")
        # Get the "Seat Assignment" (dispatch) and "Ticket Value" (combine)
        # dispatch_mask: (Tokens, Experts, Capacity) - Binary
        # combine_weights: (Tokens, Experts, Capacity) - Weighted Probabilities
        # gates for the aux_loss
        dispatch_mask, combine_weights, gates, router_z_loss = self.top2gating(x_flat)

        # 3. Dispatch (The "Sort")
        # We move tokens into the expert buffers.
        # This handles sorting, padding, and dropping all at once.
        # (Tokens, Dim) x (Tokens, Experts, Capacity) -> (Experts, Capacity, Dim)
        expert_input = torch.einsum('td, tec -> ecd', x_flat, dispatch_mask)

        # 4. Compute (The "Engine")
        # Run the experts in parallel on the sorted buffers.
        # Input: (Experts, Capacity, Dim) -> Output: (Experts, Capacity, Dim)
        expert_output = self.compute_from_dispatch(expert_input)

        weighted_output = torch.einsum('tec, ecd -> td', combine_weights, expert_output)

        # 5. Reshape
        output = weighted_output.view(batch, seq,d_model)

        # Load balancing loss gets added the Router Z-loss
        aux_loss = self.load_balancing_loss(gates, dispatch_mask)
        total_aux_loss = aux_loss +  router_z_loss


        return output , total_aux_loss



    def compute_from_dispatch(self,expert_input):
        """Implements the compute from pre dispatched tensor.
        x_packed is (E,C,D), weights is (E,C,1)"""

        # This is not needed
        # layer_1_experts = self.layer_1.view(self.n_experts, self.ffn_dim, self.d_model)
        # layer_2_experts = self.layer_2.view(self.n_experts, self.d_model, self.ffn_dim)
        
        # Take product of matrix (expert input layer->hidden layer)
        # (Experts, Capacity, D) @ (Experts, D, FFN) -> (Experts, Capacity, FFN)
        h = F.relu(torch.einsum('ECD,EDH->ECH', expert_input, self.w1))
       
        # Take product of matrices (hidden layer->output-layer) 
        # Contract over h
        output = torch.einsum('ECH, EHD -> ECD', h, self.w2)
        return output
        
        

    def top2gating(self, x_flat):
        batch_s , d = x_flat.shape


        num_tokens = batch_s
        capacity = int(num_tokens/self.n_experts * self.capacity_factor)
        capacity = max(capacity, 1)

        # (T,E) -> (T,E)
        logits = self.gates(x_flat)

        # We introduce Z-loss froom Google Sparse MoE paper
        router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        gates = F.softmax(logits, dim=-1)
    
        # loop implementation is gone (g_1_idx is (T,))
        # top2_vals: (Batch*Seq, 2)
        # top2_indices: (Batch*Seq, 2)
        
        top2_vals, top2_indices = torch.topk(gates, k=2)

        # Normalize in one line
        # We use softmax on the top-2 scores, or just normalize the raw top-2 sum
        # (N, 2) / (N, 1) -> Broadcasting handles it but we keep dim
        top2_weights = top2_vals / top2_vals.sum(dim=-1, keepdim=True)

        # Put the conditional with a cumulative sum (one-hot the g_1idx)
        # This is (T,E) (one-hot across the choices for first expert)
        g_1_idx = F.one_hot(top2_indices[:,0], num_classes=self.n_experts).int()

        # This is (T,E) (one-hot) (across the choices for second expert)
        g_2_idx = F.one_hot(top2_indices[:,1], num_classes=self.n_experts)
   
        # Sum how mauch capacity is used at token t (shape is the same)
        g_1_idx_cumsum =torch.cumsum(g_1_idx, dim=0) 

        # G_1_idx_cumsum needs to be lower than capacity and the token that selected that expert
        # Shape is (T,E) (boolean)
        mask_1 = (g_1_idx_cumsum <= capacity) & (g_1_idx> 0)

        # Counts are the sum of True, this is the final sum only.
        # (T,E) -> E
        counts_1 = mask_1.sum(dim=0)
        g_2_idx_cumsum = torch.cumsum(g_2_idx, dim=0) + counts_1

        # the other mask, similar to mask 1 
        mask_2 = (g_2_idx_cumsum <= capacity) & (g_2_idx > 0)
       
        # Transform into (token, expert, capacity) (this is one hot for each expert, which slot gets occupied)
        
        # Wo now we can write the capcity as a one-hot vector
        # And we define the slot as the cumulative effect of round 1 and round 2
        final_slot = mask_1.float() *  g_1_idx_cumsum + mask_2.float() * g_2_idx_cumsum


        # --- Dispatch Mask ---
        # Transform into (token, expert, capacity)
        # Now we have (tokens, E) let's one-hot sparsely remove 1 because slots are indices
        # Clamp because the indeces are going to -1 when the final_slot is zero (not chosen)
        # (T,E)-> (T,E,C)
        dispatch_mask = F.one_hot((final_slot - 1).clamp(min=0).long(), num_classes=capacity).float()

        # Now we are at (T, E, C) now we are moving the tokens from x_flattened
        final_mask = mask_1 | mask_2

        # This kills the "Phantom ones" created by the clamped 0s.
        # Ready to compute (T, E, C)
        dispatch_mask = dispatch_mask * final_mask.unsqueeze(-1)

        # Combine Weights
        # (T,E) by default - currently all zeros
        weights = torch.zeros(x_flat.shape[0], self.n_experts, device=x_flat.device)

        # Scatter into weights the values weights into the position
        # top2_indices
        # top_2_indices is shape (Tokens, 2)
        # top_2_values is shape (Tokens, 2)
        weights_tokens = weights.scatter(dim=1, index=top2_indices, src=top2_weights)

        # Use the dispatch mask to combine
        combine_weights = dispatch_mask * weights_tokens.unsqueeze(-1)

        return dispatch_mask, combine_weights, gates, router_z_loss
    
    def load_balancing_loss(self, gates, dispatch_mask):
        """
        Calculates the auxiliary loss to encourage uniform expert usage.
        Loss = N * sum(mean_prob_i * fraction_expert_i)
        """

        # (T, E) -> (E)
        mean_probs = gates.mean(dim=0) # Average per expert

        # 2. Hard Decisions (Discrete)
        # "How many tokens ACTUALLY went to Expert A?"
        N = self.n_experts
        #sum across tokens and capacity for the expert (Is it too concentrated in a single Expert)
        # (T, E, C) -> (E)
        fraction_expert = dispatch_mask.sum(dim=(0,2)) / dispatch_mask.shape[0]

        # Product
        loss = N * mean_probs @ fraction_expert.T
        return loss




if __name__=='__main__':
    b = 5
    d = 16
    s = 10
    n_experts = 2
    capacity_factor = 2.0 
    
    X = torch.randn(b,s,d)
    moe_layer = MoELayer(d_model=d, n_experts=n_experts, capacity_factor=capacity_factor)
    
    print("Running MoE Forward Pass...")
    output, loss = moe_layer(X)
    print("Input Shape:", X.shape)
    print("Output Shape:", output.shape)
    print("Success!")

