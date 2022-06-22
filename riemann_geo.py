import jax 
import jax.numpy as jnp

@jax.jit
def metric(coords, model=None):
    """
    Compute the metric tensor at a given coordinate.

    Parameters
    ----------
    coords : jnp.array
        Coordinates of the point at which to compute the metric.
    model :
        Function that returns the metric tensor at a given coordinate. If not specified, the Euclidean metric is used.
    
    Returns
    -------
    metric : jnp.array
        Metric tensor (two lower indices) at the given coordinate. 
    """

    if model is None:
        return jnp.eye(coords.shape[-1])
    return model(metric)

# partial derivatives of the metric
pd_metric = jax.jit(jax.jacfwd(metric))

jax.jit()
def christoffel(coords, model):
    """
    Compute the Christoffel symbols at a given coordinate.

    Parameters
    ----------
    coords : jnp.array
        Coordinates of the point at which to compute the Christoffel symbols.
    model :
        Function that returns the metric at a given coordinate.
    
    Returns
    -------
    christoffel : jnp.array
        Christoffel symbols at the given coordinate. The first index is upper, the rest two indices are lower.
    """

    met = metric(coords, model)
    inv_met = jnp.linalg.inv(met)
    partial_derivs = jnp.einsum('mns -> smn', pd_metric(coords, model))
    sum_partial_derivs = partial_derivs + jnp.einsum('nrm -> mnr', partial_derivs) - jnp.einsum('rmn -> mnr', partial_derivs)
    christ = 0.5 * jnp.einsum('sr, mnr -> smn', inv_met, sum_partial_derivs)
    return christ

# partial derivatives of the christoffel symbols
pd_christoffel = jax.jit(jax.jacfwd(christoffel))

@jax.jit
def riemann_curvature(coords, model):
    """
    Compute the Riemann curvature at a given coordinate.

    Parameters
    ----------
    coords : jnp.array
        Coordinates of the point at which to compute the Riemann curvature.
    model :
        Function that returns the metric at a given coordinate.
    
    Returns
    -------
    riemann_curvature : jnp.array
        Riemann curvature at the given coordinate. The first index is upper, the rest three indices are lower.
    """

    christ = christoffel(coords, model)
    pd_christ = jnp.einsum('rmns -> srmn', pd_christoffel(coords, model))
    return jnp.einsum('mrns -> rsmn', pd_christ) - jnp.einsum('nrms -> rsmn', pd_christ) + jnp.einsum('rml, lns -> rsmn', christ, christ) - jnp.einsum('rnl, lms -> rsmn', christ, christ)

@jax.jit
def ricci_tensor(coords, model):
    """
    Compute the Ricci tensor at a given coordinate.

    Parameters
    ----------
    coords : jnp.array
        Coordinates of the point at which to compute the Ricci tensor.
    model :
        Function that returns the metric at a given coordinate.
    
    Returns
    -------
    ricci_tensor : jnp.array
        Ricci tensor (two lower indices) at the given coordinate. 
    """

    riemann = riemann_curvature(coords, model)
    return jnp.einsum('rsru -> su', riemann)

@jax.jit
def ricci_scalar(coords, model):
    """
    Compute the Ricci scalar at a given coordinate.

    Parameters
    ----------
    coords : jnp.array
        Coordinates of the point at which to compute the Ricci scalar.
    model :
        Function that returns the metric at a given coordinate.
    
    Returns
    -------
    ricci_scalar : jnp.array
        Ricci scalar at the given coordinate. 
    """

    return jnp.einsum('mn, mn -> ', jnp.linalg.inv(metric(coords, model)), ricci_tensor(coords, model))

@jax.jit
def einstein_tensor(coords, model):
    """
    Compute the Einstein tensor at a given coordinate.

    Parameters
    ----------
    coords : jnp.array
        Coordinates of the point at which to compute the Einstein tensor.
    model :
        Function that returns the metric at a given coordinate.
    
    Returns
    -------
    einstein_tensor : jnp.array
        Einstein tensor (two lower indices) at the given coordinate. 
    """
    met = metric(coords, model)
    ricci_ts = ricci_tensor(coords, model)
    return ricci_ts - 0.5 * jnp.einsum('mn, mn -> ', jnp.linalg.inv(met), ricci_ts).reshape(1, 1) * met

