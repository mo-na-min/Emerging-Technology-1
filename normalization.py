import numpy as np

def normalize_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    Normalizes 21 hand 3D landmarks for translation and scale invariance.
    
    Args:
        landmarks_3d: Array of shape (21, 3) representing (x, y, z) coordinates.
        
    Returns:
        Flattened 1D numpy array of length 63.
    """
    pts = np.array(landmarks_3d, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected landmark shape (21, 3), got {pts.shape}")
    
    # 1. Translation Invariance: Subtract Wrist position (Landmark 0)
    wrist = pts[0].copy()
    translated = pts - wrist
    
    # 2. Scale Invariance: Divide by distance between Wrist (0) and Middle Finger MCP (9)
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    
    if scale < 1e-6:
        scale = 1.0  # Prevent division by zero if hand points collapse
        
    normalized = translated / scale
    return normalized.flatten()