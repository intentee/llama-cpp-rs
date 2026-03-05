/// The Rope type that's used within the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Standard rotary positional encoding.
    Norm,
    /// GPT-NeoX style rotary positional encoding.
    NeoX,
    /// Multi-dimensional rotary positional encoding.
    MRope,
    /// Vision model rotary positional encoding.
    Vision,
}
