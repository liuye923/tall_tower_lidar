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

class CommonPlot:
    '''Common setting for 2D-map plot and curve plot'''
    def __init__(self):
        '''initilization & load global variables'''
        pass
    def __call__(self):
        pass
    def update_global_variable(self):
        for k, v in self.kwargs.items():
            V = global_variable.get_value(k.upper())
            if V is not None: self.kwargs[k] = V
        return self.kwargs

    def add_contourf_map(self, ax, *args, **kwargs):
        '''wrapper for contourf or contour'''
        self.kwargs = look_up_table.add_map(kwargs.get('domain', 'SGP'))
        self.kwargs.update(kwargs)
        kwargs = self.update_global_variable()
        #...set axis on
        ax.set_axis_on()
        #...titie
        self.add_title_on_map('left', **kwargs)
        self.add_title_on_map('right', **kwargs)
        #...extent
        extent = kwargs.get('extent')
        if extent is not None: ax.set_extent(extent, crs=ccrs.PlateCarree())

        #...draw
        cs = self.add_contourf(ax, *args, **kwargs)

        #...draw topo
        if kwargs.get('draw_topo'): cs_to = self.add_topo(ax, *args, **kwargs)

        #...draw state boundary
        if kwargs.get('draw_state_boundary'): self.add_state_boundary(ax)

        #...draw gridlines
        if kwargs.get('draw_gridlines'): self.add_gridlines(ax, **kwargs)

        return cs

    def add_contourf(self, ax, lon, lat, data, **kwargs):
        '''draw contourf'''
        cmap = kwargs.get('cmap', 'WhiteBlueGreenYellowRed')
        clev = kwargs.get('clev')
        #...colormap
        cmap, clev, norm = self._colormap(cmap, clev, cend=kwargs.get("cend", None))
        #...draw
        contourf_kwargs = dict(levels=clev, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        cs = ax.contourf(lon, lat, data, **contourf_kwargs)
        return cs, clev

    def add_title_on_map(self, entry, **kwargs):
        if entry == 'left':
            lefts  = kwargs.get('leftstr', None)
            leftsize  = kwargs.get('leftstr_size')
            if lefts is not None:  ax.text(0.0, 1.01, lefts, size=leftsize, transform=ax.transAxes)
        if entry == 'right':
            rights = kwargs.get('rightstr', None)
            rightsize = kwargs.get('rightstr_size')
            if rights is not None: ax.text(1.0, 1.01, rights, size=rightsize, transform=ax.transAxes, horizontalalignment='right')


    def _colormap(self, cmap, clev, cend=None):
        '''extract colormap and levels'''
#        if type(cmap)==str: cmap = getattr(gvcmaps, cmap) 
        if type(cmap)==str: cmap = getattr(cmaps, cmap) 
        if cend is not None:
            print(cend)
            cmap = cmap[:cend,:]
            cmap.N = cend
        norm = matplotlib.colors.BoundaryNorm(clev, cmap.N)
        return cmap, clev, norm

    def add_topo(self, ax, *args, **kwrags):
        if kwargs.get('drawTopo', False):
            cmapTopo  = kwargs.get('cmapTopo', 'WhiteBlueGreenYellowRed') 
            clevTopo  = kwargs.get('clevTopo', [500,1000,2000])
            topoType = kwargs.get('topoType', 'contour')
            if topoType == 'contour':
                cs1 = ax.contour(etopo.lon, etopo.lat, etopo, clevTopo, linewidths=(0.8,1.5,2), colors='black', alpha=0.7, transform=ccrs.PlateCarree())
            else:
                cs1 = ax.contourf(etopo.lon, etopo.lat, etopo, clevTopo, cmap=cmapTopo, colors='black', alpha=0.7, transform=ccrs.PlateCarree())
    
    def add_state_boundary(self, ax, **kwargs):
        '''draw state boundary'''
        states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                                     name='admin_1_states_provinces_shp')
        ax.add_feature(states, linewidth=0.5, edgecolor='gray')
        ax.coastlines('50m', linewidth=0.5, zorder=10)
#        ax.add_feature(LAND,  facecolor='lightgray', alpha=0.4, zorder=0)
        ax.add_feature(LAND,  facecolor='gray', alpha=1.0, zorder=0)
        ax.add_feature(OCEAN, edgecolor='none', linewidth=0.5, alpha=0.4, facecolor='lightgray', zorder=0)

    def add_gridlines(self, ax, **kwargs):
        self._add_gridlines(ax, **kwargs)

    def _add_gridlines(self, ax, xticks=None, yticks=None, **kwargs):
        # keywards: gl_kwargs={draw_labels=True, linewidth=1, color='k', x_inline=False, y_inline=False, alpha=0.5, linestyle='--', zorder=10}
        # keywards: xlabel_size=12, ylabel_size=12, gridlines_width=1, gridlines_color='k'
        import matplotlib.ticker as mticker
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
        gl_kwargs=dict(draw_labels=True, linewidth=1, color='k', x_inline=False, y_inline=False, alpha=0.5, linestyle='--', zorder=10)
        gl_kwargs.update(kwargs.get('gl_kwargs',{}))
        gl = ax.gridlines(**gl_kwargs)
    
        gl.right_labels  = kwargs.get('right_labels', False)
        gl.bottom_labels = kwargs.get('bottom_labels', True)
        gl.left_labels   = kwargs.get('left_labels', True)
        gl.top_labels    = kwargs.get('top_labels', False)
    
        gl.xlines = kwargs.get('xlines', True)
        gl.ylines = kwargs.get('ylines', True)
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': kwargs.get('xlabel_size',12)}
        gl.ylabel_style = {'size': kwargs.get('ylabel_size',12)}
        gl.rotate_labels = False
        return gl

myplot = CommonPlot()
