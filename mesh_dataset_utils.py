import numpy as np
import trimesh
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import robust_laplacian
import os

# ======================================
# ========= SPECTRAL EMBEDDING =========
# ======================================

# gr : nx Graph
# k : k-largest eigenvalues

# returns:
# ndarray with shape (k,)
# k eigenvalues

# ndarray with shape (N, k)
# for N = number of nodes in gr 
#     k = number of eigenvalues 
def laplacian_eigenmap(gr, k, lap=None):
    
    """Creates eigen values and vectors of a graph
    
    Calculates the laplacian matrix of a graph and returns the eigen vectors and eigen values
    
    Args:
        gr (networkX graph): current graph
        k (int): number of largest eigen values
    
    Returns:
        numpy array (shape: (k,)): eigen values
        numpy array (shape: (N, k)): eigen vectors 
        (N = # of nodes in gr, k = # of eigen values)"""
    
    lap = nx.laplacian_matrix(gr)
    lap = lap.asfptype()
    eigs = scipy.sparse.linalg.eigsh(lap, k=k)
    e_vals = eigs[0]
    e_vecs = eigs[1]
    
    return e_vals, e_vecs


def mesh_laplacian_eigenmap(verts, faces, k, lap=None):
    
    """Creates eigen values and vectors of a mesh
    
    Calculates the laplacian matrix of a mesh and returns the eigen vectors and eigen values
    
    Args:
        verts (numpy array): array of the vertices of a mesh
        faces (numpy array): matrix of the faces of a mesh
        k (int): number of eigen values
    
    Returns:
        numpy array (shape: (k,): eigen values
        numpy array (shape: (N, k)): eigen vectors"""
    
    L, M = robust_laplacian.mesh_laplacian(verts, faces)
    # L = L.asfptype()
    eigs = scipy.sparse.linalg.eigsh(L, k, M)
    e_vals = eigs[0]
    e_vecs = eigs[1]
    
    return e_vals, e_vecs

# ======================================
# ========= POINTWISE OPERATIONS =======
# ======================================

def get_dists(obj1, obj2):
    
    """Finds the normal of offsets of each of the vertices between 2 meshes
    
    Determines whether or not the 2 objects have the same shape and then calculates the normal of
    the difference of the 1st and 2nd object's vertices. Then returns these offsets.
    
    Args:
        obj1 (mesh): mesh of the 1st object
        obj2 (mesh): mesh of the 2nd object
    
    Returns:
        numpy array (shape: (V, k)): array of distances between all vertices 
        (V = # of Vertices, k = shape of normal, most likely 1)"""
    
    if obj1.vertices.shape[0] != obj2.vertices.shape[0]:
        raise Exception('objects are not one-to-one')
    
    offsets = np.zeros(obj1.vertices.shape[0])
    for i, face in enumerate(obj1.vertices):
        offsets[i] = np.linalg.norm(obj1.vertices[i]-obj2.vertices[i])
    return offsets

def get_offsets(obj1, obj2):
    
    """Finds the offsets of each of the vertices between 2 meshes
    
    Determines whether or not the 2 objects have the same shape and then calculates
    the difference of the 1st and 2nd object's vertices. Then returns the offsets.
    
    Args:
        obj1 (mesh): mesh of the 1st object
        obj2 (mesh): mesh of the 2nd object
    
    Returns:
        numpy array (shape: (V, k)): array of distances between all vertices 
        (V = # of Vertices, k = shape of a vertice)"""
    
    if obj1.vertices.shape[0] != obj2.vertices.shape[0]:
        raise Exception('objects are not one-to-one')
    
    offsets = np.zeros((obj1.vertices.shape[0], 3))
    for i, face in enumerate(obj1.vertices):
        offsets[i] = obj1.vertices[i]-obj2.vertices[i]
    return offsets


def get_transform(x1, x2):
    
    """Computes the linear transformation that maps x1 to x2

    Args:
        x1 (np.array) input vector
        x2 (np.array) transformed vector

    Returns:
        A (np.array) Array representing the linear transformation from x1 to x2

    """
    
    assert x1.shape[0] == 6
    assert x1.shape[0] == 6

    X1 = np.vstack([x1])
    X2 = np.vstack([x2])
    
    A, residuals, rank, s = np.linalg.lstsq(X1, X2, rcond=None)
    return A.reshape((36,))


def get_linear_transformations(smooth, orig):
    
    """Gets surface normal transformation targets from smoothed mesh to original

    Calculates the surface normal for each vertex on the smooth mesh and its corresponding normal on the original mesh.
    Then calculates the linear transformation matrix that transforms the smoothed normal to the original normal.
    Finally, linear transformation matrices are flattened.

    Args:
        smooth (trimesh mesh) smoothed mesh
        orig (trimesh mesh) original mesh
    Returns:
        lin_trans (np.array) array of shape (V, 9) where V is the number of vertices in the mesh, and each entry
            is the flattened linear transformation matrix of surface normals for the corresponding pair of vertices.
    """
    
    smooth_sns = surface_normals = np.concatenate((smooth.vertex_normals, smooth.vertices), axis=1)
    orig_sns = surface_normals = np.concatenate((orig.vertex_normals, orig.vertices), axis=1)

    all_lin_trans = np.zeros((smooth_sns.shape[0], 36))
    for i in range(smooth_sns.shape[0]):
        all_lin_trans[i,:] = get_transform(smooth_sns[i,:], orig_sns[i,:])
    
    return all_lin_trans

# ======================================
# ============= SMOOTHING ==============
# ======================================

# smoothes a mesh using trimesh laplacian smoothing
def smooth_mesh_lap(mesh, iterations):
    
    """Creates a smooth mesh using laplacian smoothing
    
    Applies the laplacian smoothing algorithm to a given mesh based on a given number of iterations
    
    Args:
        mesh (mesh): mesh of current object
        iterations (int): number of times smoothing algorithm is applied to object
    
    Returns:
        mesh: smoothed mesh
        mesh: original mesh"""
    
    orig = mesh
    smooth = mesh.copy()
    smooth = trimesh.smoothing.filter_laplacian(smooth, iterations=iterations)
    # print("laplacian")
    return smooth, orig

# smoothes a mesh using mutable diffusion laplacian smoothing
def smooth_mesh_mut_dif(mesh, iterations):
    
    """Creates a smooth mesh using mutable difference smoothing
    
    Applies the mutable difference smoothing algorithm to a given mesh based on a given number of iterations
    
    Args:
        mesh (mesh): mesh of current object
        iterations (int): number of times smoothing algorithm is applied to object
    
    Returns:
        mesh: smoothed mesh
        mesh: original mesh"""
    
    orig = mesh
    smooth = mesh.copy()
    smooth = trimesh.smoothing.filter_mut_dif_laplacian(smooth, iterations=iterations)
    # print("mut dif")
    return smooth, orig

#smoothes a mesh using taubin smoothing
def smooth_mesh_taubin(mesh, iterations):
    
    """Creates a smooth mesh using taubin smoothing
    
    Applies the taubin smoothing algorithm to a given mesh based on a given number of iterations
    
    Args:
        mesh (mesh): mesh of current object
        iterations (int): number of times smoothing algorithm is applied to object
    
    Returns:
        mesh: smoothed mesh
        mesh: original mesh"""
    
    orig = mesh
    smooth = mesh.copy()
    smooth = trimesh.smoothing.filter_taubin(smooth, iterations=iterations)
    # print("taubin")

    return smooth, orig

def smooth_mesh(mesh, iterations, type='lap'):
    
    """Chooses an algorithm to smooth mesh
    
    Returns the result of a smoothing algorithm (chosen based on the variable 'type') after
    a certain number of iterations
    
    Args:
        mesh (mesh): mesh of current object
        iterations (int): number of times smoothing algorithm is applied to object
        type (string): type of smoothing algorithm used on mesh
    
    Returns:
        mesh: result of chosen algorithm
        mesh: original mesh"""
    
    if type == 'lap':
        return smooth_mesh_lap(mesh, iterations)
    elif type == 'taubin':
        return smooth_mesh_taubin(mesh, iterations)
    elif type == 'mut_dif':
        return smooth_mesh_mut_dif(mesh, iterations)
    else:
        raise Exception(f'{type} not valid smoothing method. Choose from: lap, taubin, mut_dif')



# ======================================
# ========= DATASET BUILDERS ===========
# ======================================


def build_offset_dataset(mesh, smooth_iter=200, lap_type='mesh', smooth_type='lap'):
    
    """Creates a dataset for offsets between original mesh and the smoothed mesh
    
    Smooths the given mesh based on a given number of iterations and then calculates the offsets of 
    these meshes. Then, calculates the eigen values and vectors of the smooth mesh and 
    places the data in a dictionary divided into fouriers, points, and targets. 
    Returns the dataset as well as the smoothed mesh
    
    Args:
        mesh (mesh): mesh of the current object
        smooth_iter (int): number of times smoothing algorithm is applied to mesh
        lap_type (string): type of laplacian for eigen data
        smooth_type (string): type of smoothing algorithm used on mesh
    
    Returns:
        dictionary: dictionary of all data of mesh (eigen vectors, vertices of smoothed mesh, 
        offsets between original and smoothed mesh)
        mesh: smoothed mesh of original mesh"""

    smooth, orig = smooth_mesh(mesh, smooth_iter, type=smooth_type)
    offsets = get_offsets(smooth, orig)
    
    if lap_type == 'mesh':
        e_vals, e_vecs = mesh_laplacian_eigenmap(np.array(smooth.vertices), np.array(smooth.faces), 100)
    else:
        # note that laplacian_eigenmap implementation is very inneficient at the moment. 
        # Pretty sure the slowness comes from using nx laplacian calculation which is really slow on larger graphs.
        gr = smooth.vertex_adjacency_graph
        e_vals, e_vecs = laplacian_eigenmap(gr, 100)
    
    dataset = {
        'fourier' : e_vecs,
        'points' : smooth.vertices,
        'target' : offsets      
          }
    return dataset, smooth

def build_norms_dataset(mesh, smooth_iter=200, lap_type='mesh', smooth_type='lap'):
    
    """Creates a dataset for the normal of offsets between original mesh and the smoothed mesh

    Smooths the given mesh based on a given number of iterations and then calculates the linear
    transformation between the original and smooth mesh. Then, calculates the eigen values and 
    vectors of the smooth mesh and places the data in a dictionary divided into fouriers, points, 
    and targets. Returns the dataset as well as the smoothed mesh

    Args:
        mesh (mesh): mesh of the current object
        smooth_iter (int): number of times smoothing algorithm is applied to mesh
        lap_type (string): type of laplacian for eigen data
        smooth_type (string): type of smoothing algorithm used on mesh

    Returns:
        dictionary: dictionary of all data of mesh (eigen vectors, normals and vertices of smoothed mesh, 
        targets of surface normals)
        mesh: smoothed mesh of original mesh"""
    
    smooth, orig = smooth_mesh(mesh, smooth_iter, type=smooth_type)
    lin_trans = get_linear_transformations(smooth, orig)
    
    if lap_type == 'mesh':
        e_vals, e_vecs = mesh_laplacian_eigenmap(np.array(smooth.vertices), np.array(smooth.faces), 100)
    else:
        # note that laplacian_eigenmap implementation is very inneficient at the moment. 
        # Pretty sure the slowness comes from using nx laplacian calculation which is really slow on larger graphs.
        gr = smooth.vertex_adjacency_graph
        e_vals, e_vecs = laplacian_eigenmap(gr, 100)
    
    dataset = {
        'fourier' : e_vecs,
        'points' : np.concatenate((smooth.vertex_normals, smooth.vertices), axis=1),
        'faces': smooth.faces,
        'target' : lin_trans      
          }
    return dataset, smooth

def build_time_dataset(mesh, smooth_iter, lap_type='mesh', smooth_type='lap', sample_size=.20, type_of_offset='norm'):
    
    """Creates a dataset for the normal of offsets between original mesh and the smoothed mesh

    Smooths the given mesh based on a given number of iterations and then calculates the linear
    transformation between the original and smooth mesh. Then, calculates the eigen values and 
    vectors of the smooth mesh and places the data in a dictionary divided into fouriers, points, 
    and targets. Takes the sample of the dataset before returning the dataset as well as the smoothed mesh

    Args:
        mesh (mesh): mesh of the current object
        smooth_iter (int): number of times smoothing algorithm is applied to mesh
        lap_type (string): type of laplacian for eigen data
        smooth_type (string): type of smoothing algorithm used on mesh
        sample_size (float): size of sample

    Returns:
        dictionary: dictionary of a sample dataset of the mesh (eigen vectors, normals and vertices of smoothed mesh, 
        targets of surface normals)
        mesh: smoothed mesh of original mesh"""
    
    dataset={}
    if(type_of_offset=='norm'):
        #print("inside1")
        dataset, smooth=build_norms_dataset(mesh, smooth_iter, lap_type, smooth_type)
    else:
        #print("inside")
        dataset, smooth=build_offset_dataset(mesh, smooth_iter, lap_type, smooth_type)
    #picks indices from the dataset (assuming that len(fouriers)=len(targets))
    
    sample_indices=np.random.sample(range(0, len(dataset['fourier']), len(dataset['fourier']*sample_size)
    dataset={
        'fourier':[dataset['fourier'][i] for i in sample_indices],
        'targets':[dataset['target'][i] for i in sample_indices]
    }
    return dataset, smooth


def build_blank_dataset(blank, DIR_NAME, lap_type='mesh'):
    
    """Creates a blank dataset by collecting data from an already smooth mesh
    
    Creates a dataset for a untextured ball by calculating the eigen values and eigen vectors and places
    the data in dictionary divied into fouriers, points, and targets. Then, creates a directory to store the
    data and saves it in that directory. Returns the file.
    
    Args:
        blank (mesh): smooth mesh
        DIR_NAME (string): directory to save dataset
        lap_type (string): type of laplacian for eigen data
    
    Returns:
        file: location of dataset"""
    
    if lap_type == 'mesh':
        e_vals, e_vecs = mesh_laplacian_eigenmap(np.array(blank.vertices), np.array(blank.faces), 100)
    else:
        gr = blank.vertex_adjacency_graph
        e_vals, e_vecs = laplacian_eigenmap(gr, 100)
    
    dataset = {
    'fourier' : e_vecs,
    'target' : np.zeros(blank.vertices.shape[0]),
    'points' : blank.vertices   
          }
    
    if not os.path.exists(f'{DIR_NAME}/npz_files'):
        os.makedirs(f'{DIR_NAME}/npz_files')
    np.savez(f'{DIR_NAME}/npz_files/{DIR_NAME}', fourier=dataset['fourier'], points=dataset['points'], target=dataset['target'])
    
    return f'/Users/maxperozek/GINR-texture/{DIR_NAME}'

# ======================================
# ========== INFERENCE MESH ============
# ======================================

def get_sphere(size=0):
    
    """Creates blank sphere
    
    Creates a smooth sphere with a given number of subdivisions
    
    Args:
        size (int): number of times subdivision is applied
    
    Returns:
        mesh: blank sphere with subdivision"""
    
    
    sphere = trimesh.primitives.Sphere()
    if size > 0:
        new_verts, new_faces = trimesh.remesh.subdivide_loop(sphere.vertices, sphere.faces, iterations=size)
        sphere = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
    return sphere


# ======================================
# ========= APPLY INFERENCE ============
# ======================================


# blank: trimesh mesh object
# offset_preds: np array of model predictions. shape (N,3) where N=number of vertices in blank mesh
def generate_pred_mesh(offset_preds, blank):
    
    """Creates new predicted mesh based on predicted offsets on a smooth sphere
    
    Places predicted offsets onto the untextured ball by adding them onto the ball's vertices
    and repairs the new mesh. Returns the new predicted mesh
    
    Args:
        offset_preds (numpy array): predicted offsets from GINR model
        blank (mesh): smooth sphere
    
    Returns:
        mesh: predicted mesh based on predicted offsets"""
    
    pred_verts =  blank.vertices - offset_preds
    inference = trimesh.Trimesh(vertices=pred_verts, faces=blank.faces)
    return inference
