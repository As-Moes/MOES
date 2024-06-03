
import numpy as np
    
def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    u_vec1 = vec1 / np.linalg.norm(vec1)  # unit vector
    u_vec2 = vec2 / np.linalg.norm(vec2)
    cos = np.dot(u_vec1, u_vec2)
    return np.arccos(cos)

 
