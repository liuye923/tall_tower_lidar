import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from streamplot import streamplot

def sample_data(shape=(20, 30)):
    """
    Return ``(x, y, u, v, crs)`` of some vector data
    computed mathematically. The returned crs will be a rotated
    pole CRS, meaning that the vectors will be unevenly spaced in
    regular PlateCarree space.

    """
    crs = ccrs.PlateCarree()

    x = np.linspace(30.9, 150.1, shape[1])
    y = np.linspace(-23.6, 24.8, shape[0])

    x2d, y2d = np.meshgrid(x, y)
    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
    v = 20 * np.cos(6 * np.deg2rad(x2d))

    return x, y, u, v, crs

#w = 3
#Y = np.linspace(-60,60,16)
#X = np.linspace(-30,30,16)
#x, y = np.meshgrid(X, Y)
#print(y)
#print(x)
##w = 3
##Y, X = np.mgrid[-w:w:8j, -w:w:8j]
#
#U = -x/10
#V = y/10

x, y, u, v, vector_crs = sample_data(shape=(20, 30))
magnitude = (u ** 2 + v ** 2) ** 0.5
print(x.shape)
print(y.shape)
print(u.shape)

def curlyvector(ax, X, Y, U, V, minlength=None, density=None, scale=0.2):
    U0 = U.copy()
    V0 = V.copy()
    norm0 = np.sqrt(U**2 + V**2)
    
    grains = 1
    U = U * grains
    V = V * grains

#    if density is not None:
#        if np.isscalar(density):
#            if density <= 0:
#                raise ValueError("If a scalar, 'density' must be positive")
#            U = U[::density, ::density]
#            V = V[::density, ::density]
#            X = X[::density, ::density]
#            Y = Y[::density, ::density]
#        else:
#            if len(density) != 2:
#                raise ValueError("'density' can have at maximum 2 dimensions")
#            U = U[::density[0], ::density[1]]
#            V = V[::density[0], ::density[1]]
#            X = X[::density[0], ::density[1]]
#            Y = Y[::density[0], ::density[1]]

    norm = np.sqrt(U**2 + V**2)
    norm_flat = norm.flatten()
    
    print(X)
    print(Y)
    X, Y = np.meshgrid(X, Y)
    start_points = np.array([X.flatten(),Y.flatten()]).T
    print(start_points.shape)

    scale = scale/np.max(norm) 

    plt.title('scaling only the length')
    for i in range(start_points.shape[0]):
        print(i)
        if minlength is not None: 
            if norm_flat[i] < minlength*grains**2: continue
        print(X)
        streamplot(ax, X, Y, U, V, 
            color='k', 
            start_points=np.array([start_points[i,:]]),
#            minlength=.95*norm_flat[i]*scale, 
#            maxlength=1.0*norm_flat[i]*scale,
            integration_direction='backward', 
            density=10, 
            arrowsize=0.0, 
            transform=ccrs.PlateCarree())
    if minlength is not None:
        U = np.where(norm>minlength, U, np.nan)
        V = np.where(norm>minlength, V, np.nan)
    plt.quiver(X, Y, U/norm, V/norm, scale=30)
    cs = ax.contourf(X,Y, norm0, cmap=plt.cm.tab20, alpha=0.5, zorder=-1, transform=ccrs.PlateCarree())
    plt.colorbar(cs, ax=ax)

    plt.show()

fig, ax = plt.subplots(1,1,figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
curlyvector(ax, x, y, u, v, minlength=2.4, scale=1.0)
#transform=ccrs.PlateCarree()
