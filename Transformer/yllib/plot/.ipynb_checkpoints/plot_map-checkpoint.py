from .importlibs import *
from . import look_up_table 
from . import global_variable
#import wrf
from shapely.geometry.polygon import Polygon
from cartopy import crs as ccrs
from cartopy.feature import NaturalEarthFeature, OCEAN, LAND, LAKES
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmaps
import cartopy.crs as ccrs 
import matplotlib.ticker as mticker
#from geocat.viz import util as gvutil
#from geocat.viz import cmaps as gvcmaps
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
from .vectorplot import velovect


class CommonPlot:
    '''Common setting for 2D-map plot and curve plot'''
    def __init__(self):
        '''initilization & load global variables'''
        pass
    def __call__(self):
        pass
   
    def set_map(self, ax, **kwargs):
        import matplotlib.path as mpath
        '''set map extent, grid'''
        #...titie
        if (extent:=kwargs.get("extent")) is not None: ax.set_extent(extent, crs=ccrs.PlateCarree())
        #...draw state boundary
        if kwargs.get("draw_state_boundary", True): 
            self.add_state_boundary(ax, **kwargs.get("boundary_kw",{}))
        #...draw gridlines
        if kwargs.get("draw_gridlines", True): 
            self.add_gridlines(ax, extent=extent, **kwargs.get("gridlines_kw",{}))
            self.add_gridlabels(ax, extent=extent, **kwargs.get("gridlabels_kw",{}))

    def add_geolocation(self, ax, point, **_kwargs):
        kwargs = dict(linestyle='', marker='o', mfc='r', mec='r', markersize=6, transform=ccrs.Geodetic())
        kwargs.update(_kwargs)
        ax.plot(*point, **kwargs)

    def add_colorbar(self, cs, ax, ticks, units=None, **_kwargs):
        orientation = _kwargs.get('orientation', 'horizontal')
        if orientation == 'horizontal': panchor = (0.5, 0.5)
        if orientation == 'vertical': panchor = (1.0, 0.0)
        print(panchor)
        kwargs = dict(ax=ax,
                      orientation=orientation,
                      ticks=ticks,
#                      drawedges=True,
                      extendrect=True,
                      # panchor=panchor,
                      pad=0.08,
                      shrink=0.75,
                      aspect=30,
                      fraction=0.05,
                     )
        kwargs.update(_kwargs)
        ccs = plt.colorbar(cs, **kwargs)
        if units is not None:
            cax = ccs.ax
            if kwargs['orientation']=='horizontal':
                cax.text(1.01, 0.5, units, transform=cax.transAxes, va="center", fontsize=12) 
            if kwargs['orientation']=='vertical':
                cax.text(0.01, 1.05, units, transform=cax.transAxes, ha="left", fontsize=12) 
        return ccs

    #...add vector overlay to plot
    def add_vector(self, ax, lon, lat, u, v, **kwargs):
        if not kwargs.get('nomap', False):
            vector_default_kwargs = dict(
                pivot='tail', width=0.003, 
                scale_units="inches", scale=300,
                headwidth=5, alpha=1,
                transform=ccrs.PlateCarree()
            )
        else:
            vector_default_kwargs = dict(
                pivot='tail', width=0.003, 
                scale_units="inches", scale=300,
                headwidth=5, alpha=1
            )
        if 'nomap' in kwargs: del kwargs['nomap']
        vector_default_kwargs.update(kwargs.get('vector_kwargs', {}))
        cs = ax.quiver(lon, lat, u, v, **vector_default_kwargs)
#        grains = kwargs.get("grains", 10)
#        scale  = kwargs.get("scale", 1.0)
#        linewidth = kwargs.get("linewidth", 0.8)
#        color  = kwargs.get("color", "k")
#        density = kwargs.get("density", 1)
#        minlength = kwargs.get("minlength")
#        cs = velovect(ax, lon, lat, u.data, v.data, 
#            arrowstyle='fancy', 
#            scale=scale, grains=grains, 
#            color=color, 
#            linewidth=linewidth,
#            density=density,
#            minlength=minlength,
#            transform=ccrs.PlateCarree()
#        )
        return cs

    def add_quiverkey(self, ax, vplot, x=0.83, y=-0.16, length=8, string='8 $ms^{-1}$', **kwargs):
        quiverkey_default_kwargs = dict(
            labelpos='W',
            coordinates='axes',
            color='black'
        )
        quiverkey_default_kwargs.update(kwargs)
        qk = ax.quiverkey(vplot, x, y, length, string, **quiverkey_default_kwargs)


    def add_hatches(self, ax, *args, **kwargs):
        '''wrapper for contourf or contour'''
        self.kwargs = look_up_table.add_map()
        self.kwargs.update(kwargs)
        ax.set_axis_on()
        #...colormap
        clev = kwargs.get('clev')
        hatches = kwargs.get("hatches", ["///"])
        plt.rcParams['hatch.color'] = kwargs.get("color","k")
        if not kwargs.get('nomap', False):
            contourf_kwargs = dict(hatches=hatches, levels=clev, colors="none", transform=ccrs.PlateCarree())
        else:
            contourf_kwargs = dict(hatches=hatches, levels=clev, colors="none")
        if 'nomap' in kwargs: del kwargs['nomap']
        cs = ax.contourf(*args, **contourf_kwargs)
        return cs, clev

    def add_contour(self, ax, *args, **kwargs):
        '''wrapper for contourf or contour'''
        self.kwargs = look_up_table.add_map()
        self.kwargs.update(kwargs)
        ax.set_axis_on()
        #...colormap
        cmap = kwargs.get('cmap', 'WhiteBlueGreenYellowRed')
        clev = kwargs.get('clev')
        colors = kwargs.get("colors", None)
        linewidths = kwargs.get("linewidths", 1.5)
        alpha = kwargs.get("alpha", 1)
#        cmap, clev, norm = self._colormap(cmap, clev, cend=kwargs.get("cend", None))
        #...draw
        contourf_kwargs = dict(levels=clev, linewidths=linewidths, colors=colors, alpha=alpha, transform=ccrs.PlateCarree())
        cs = ax.contour(*args, **contourf_kwargs)
        return cs, clev

    def add_contourf(self, ax, *args, **kwargs):
        '''wrapper for contourf or contour'''
        self.kwargs = look_up_table.add_map()
        self.kwargs.update(kwargs)
        ax.set_axis_on()
        #...colormap
        cmap = kwargs.get('cmap', 'WhiteBlueGreenYellowRed')
        clev = kwargs.get('clev')
        cmap, clev, norm = self._colormap(cmap, clev, cend=kwargs.get("cend", None))
        #...draw
        if not kwargs.get("nomap", False):
            contourf_kwargs = dict(levels=clev, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            contourf_kwargs = dict(levels=clev, cmap=cmap, norm=norm)
        cs = ax.contourf(*args, **contourf_kwargs)
        return cs, clev

    def add_pcolormesh(self, ax, *args, **kwargs):
        '''wrapper for contourf or contour'''
        self.kwargs = look_up_table.add_map()
        self.kwargs.update(kwargs)
        ax.set_axis_on()
        #...colormap
        cmap = kwargs.get('cmap', 'WhiteBlueGreenYellowRed')
        clev = kwargs.get('clev')
        cmap, clev, norm = self._colormap(cmap, clev, cend=kwargs.get("cend", None))
        #...draw
        if not kwargs.get("nomap", False):
            contourf_kwargs = dict(cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            contourf_kwargs = dict(cmap=cmap, norm=norm)
        cs = ax.pcolormesh(*args, **contourf_kwargs)
        return cs, clev

    def add_title_on_map(self, x=None, y=None, string=None, pos=None, **_kwargs):
        kwargs = dict(size=12, transform=ax.transAxes)
        kwargs.update(_kwargs)
        if pos == "left":
            if x is None: x = 0.
            if y is None: y = 1.01
        if pos == "right":
            if x is None: x = 1.
            if y is None: y = 1.01
            kwargs["horizontalalignment"] = "right"
        ax.text(x, y, string, **kwargs)
        
    def _list_to_colormap(self, cmap):
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        return ListedColormap(cmap)
    
    def _colormap(self, cmap, clev, cend=None):
        '''extract colormap and levels'''
        norm = None
        if isinstance(cmap, list): cmap = self._list_to_colormap(cmap)
        if isinstance(cmap, str):
    #        if type(cmap)==str: cmap = getattr(gvcmaps, cmap) 
            if type(cmap)==str: cmap = getattr(cmaps, cmap) 
            if cend is not None:
                print(cend)
                cmap = cmap[:cend,:]
                cmap.N = cend
            norm = matplotlib.colors.BoundaryNorm(clev, cmap.N)
        return cmap, clev, norm

    def add_state_boundary(self, ax, **kwargs):
        '''draw state boundary'''
        
        if kwargs.get('draw_state', True):
            states = NaturalEarthFeature(
                category='cultural', scale='10m', facecolor='none',
                name='admin_1_states_provinces'
            )
            state_default_kwargs = dict(linewidth=0.5, zorder=10, alpha=0.2)
            state_default_kwargs.update(kwargs.get('state_kw', {}))
            ax.add_feature(states, **state_default_kwargs)
        if kwargs.get('draw_coastline', True):
            coastline_default_kwargs = dict(linewidth=0.5, zorder=10, alpha=0.2)
            coastline_default_kwargs.update(kwargs.get('coastline_kw', {}))
            ax.coastlines('10m', **coastline_default_kwargs)
        ax.add_feature(LAND,  edgecolor='none', alpha=0.2, facecolor='none', zorder=0)
        ax.add_feature(OCEAN, edgecolor='none', alpha=0.4, facecolor='lightgray', zorder=0)

    def add_gridlines(self, ax, xticks=None, yticks=None, extent=None, **_kwargs):
        if xticks is None: xticks = np.linspace(extent[0], extent[1], 4)
        if yticks is None: yticks = np.linspace(extent[2], extent[3], 4)
        self.gridline_xticks = xticks
        self.gridline_yticks = yticks
        kwargs=dict(draw_labels=False, linewidth=1, color='k', x_inline=False, y_inline=False, alpha=0.5, linestyle='--', zorder=10)
        kwargs.update(_kwargs)
        gl = ax.gridlines(**kwargs)
        gl.xlines = kwargs.get('xlines', True)
        gl.ylines = kwargs.get('ylines', True)
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        return gl
    
    def add_gridlabels(self, ax, xticks=None, yticks=None, **kwargs):
        if xticks is None: xticks = self.gridline_xticks
        if yticks is None: yticks = self.gridline_yticks
        gl = ax.gridlines(draw_labels=False)
        gl.xlines = kwargs.get('xlines', False)
        gl.ylines = kwargs.get('ylines', False)
        gl.right_labels  = kwargs.get('right_labels', False)
        gl.bottom_labels = kwargs.get('bottom_labels',True)
        gl.left_labels   = kwargs.get('left_labels', True)
        gl.top_labels    = kwargs.get('top_labels', False)
    
        gl.xlabel_style = {'size': kwargs.get('xlabel_size',12)}
        gl.ylabel_style = {'size': kwargs.get('ylabel_size',12)}
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.rotate_labels = False
        return gl

    def add_topo(self, ax, *args, **kwrags):
        if kwargs.get('drawTopo', False):
            cmapTopo  = kwargs.get('cmapTopo', 'WhiteBlueGreenYellowRed') 
            clevTopo  = kwargs.get('clevTopo', [500,1000,2000])
            topoType = kwargs.get('topoType', 'contour')
            if topoType == 'contour':
                cs1 = ax.contour(etopo.lon, etopo.lat, etopo, clevTopo, linewidths=(0.8,1.5,2), colors='black', alpha=0.7, transform=ccrs.PlateCarree())
            else:
                cs1 = ax.contourf(etopo.lon, etopo.lat, etopo, clevTopo, cmap=cmapTopo, colors='black', alpha=0.7, transform=ccrs.PlateCarree())
    
    def add_polygon(self, ax, lats, lons, **kwargs):
        from cartopy import crs as ccrs
        default_kwargs = {
            'facecolor':'None',
            'edgecolor':'tab:blue',
            'linewidth':3,
            'zorder':10}
        default_kwargs.update(kwargs)
        pgon = self.create_polygon(lats, lons)
        ax.add_geometries([pgon], crs=ccrs.PlateCarree(), **default_kwargs)
        
    def create_polygon(self, lats, lons):
        pgon = []
        for ilat, ilon in zip(lats[:,0],     lons[:,0]    ): pgon.append([ilon.data, ilat.data])
        for ilat, ilon in zip(lats[-1,:],    lons[-1,:]   ): pgon.append([ilon.data, ilat.data])
        for ilat, ilon in zip(lats[::-1,-1], lons[::-1,-1]): pgon.append([ilon.data, ilat.data])
        for ilat, ilon in zip(lats[0,::-1],  lons[0,::-1] ): pgon.append([ilon.data, ilat.data])
        return(Polygon(pgon))
    
    def add_rectangle(self, ax, rect, **kwargs):
        rect_kwargs = dict(
            xy=(rect[0], rect[2]), width=rect[1]-rect[0], height=rect[3]-rect[2],
            ec="tab:purple", lw=2,
            fc="none", transform=ccrs.PlateCarree()
        )
        rect_kwargs.update(kwargs)
        ax.add_patch(mpatches.Rectangle(**rect_kwargs))
    
plot_map = CommonPlot()
