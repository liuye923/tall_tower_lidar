import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
from streamplot import streamplot
import cartopy.crs as ccrs
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
#    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
#    v = 20 * np.cos(6 * np.deg2rad(x2d))
    u = -y2d / 10
    v = x2d / 10

    return x, y, u, v, crs

x0, y0, U, V, crs = sample_data(shape=(100,80))
X, Y = np.meshgrid(x0, y0)

speed = np.sqrt(U*U + V*V)
print(X.shape)
print(Y.shape)
print(speed.shape)

fig, ax = plt.subplots(1,1,figsize=(16,8), subplot_kw={'projection': ccrs.PlateCarree()})

grains = 20
tmp = tuple([x]*grains for x in np.linspace(x0[0], x0[-1], grains))
xs = []
for x in tmp:
    xs += x
ys = tuple(np.linspace(y0[0], y0[-1], grains))*grains

#xs = list(x)
#ys = list(y)
print(len(xs))
print(len(ys))
seed_points = np.array([list(xs), list(ys)]).T
print(seed_points.shape)
#print(x)
#seed_points = np.array([X.flatten(),Y.flatten()]).T
#seed_points = np.array([x,y])
# Varying color along a streamline

#strm = streamplot(ax, X, Y, U, V, 
#           color="k", 
#           density=10,
#           minlength=0.001, arrowstyle='fancy',
#           maxlength=0.07,
#           integration_direction='forward', start_points = seed_points.T,
#           transform=ccrs.PlateCarree())
norm_flat = speed.flatten() / speed.max() * 0.1
minlength = 12 / speed.max() * 0.001
for i, _ in enumerate(seed_points):
    strm = streamplot(ax, X, Y, U, V, 
               color="k", 
               density=10,
               minlength=minlength, 
               maxlength=norm_flat[i],
               start_points=np.array([seed_points[i,:]]),
               integration_direction='forward', 
               arrowstyle='fancy',
               transform=ccrs.PlateCarree())
#fig.colorbar(strm.lines)
#cs = ax.contourf(X, Y, speed, cmap=plt.cm.tab20, alpha=0.5, zorder=-1, transform=ccrs.PlateCarree())
cs = ax.pcolormesh(X, Y, speed, cmap=plt.cm.tab20, alpha=0.5, zorder=-1, transform=ccrs.PlateCarree())
ax.set_title('Varying Color')
ax.coastlines()
plt.colorbar(cs, ax=ax)

plt.tight_layout()
plt.show()
