from yllib.plot import plot_map
import matplotlib as mpl
import numpy as np

def plot_cross_section(data, u=None, v=None, **kwargs):
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.family'] = 'Helvetica'  
    
    # create a figure
    if (fig:=kwargs.get('fig')) is None:  
        fig = plt.figure(figsize=(8, 5))
    if (ax:=kwargs.get('ax')) is None:  ax = plt.axes()
    ax.set_axis_on()
    
    # # set map - extent, grid, coastline ...
    # draw_xlabel = kwargs.get('draw_xlabel', True)
    # draw_ylabel = kwargs.get('draw_ylabel', True)

    # plot_map.set_map(
    #     ax,
    #     extent=kwargs.get('extent',[-125+360,-70+360,24,50]),
    #     gridlines_kw=dict(
    #         xticks=[-126,-124,-122,-120,-118,-116,-114,-112],
    #         yticks=[40,42,44,46,48,50],
    #         linestyle=":", alpha=0.5,
    #     ),
    #     gridlabels_kw=dict(
    #         xticks=[-126,-124,-122,-120,-118,-116,-114,-112] if draw_xlabel else [],
    #         yticks=[40,42,44,46,48,50] if draw_ylabel else [],
    #     ),
    #     boundary_kw=dict(
    #         draw_state=True,
    #         coastline_kw=dict(linewidth=0.9, alpha=0.7, color='k'),
    #         draw_borders=True,
    #         border_kw=dict(linewidth=0.9, alpha=0.7, color='k'),
    #     ),
    # )
    
    # lat and lon for contour plot
    if ((lon:=kwargs.get('lon')) is None): lon = data['lon']
    if ((lat:=kwargs.get('lat')) is None): lat = data['lat']
        
    # draw contourf
    cmap = kwargs.get('cmap', "WhiteBlueGreenYellowRed")
    clev = kwargs.get('clev', np.arange(0, 40, 1))

    # cs, clev = plot_map.add_contourf(
    cs, clev = plot_map.add_pcolormesh(
        ax, lon, lat, data,
        cmap=cmap,clev=clev,
        nomap=True,
    )
    # cs = ax.contourf(lon, lat, data, cmap=cm_wind['cmap'], norm=cm_wind['norm'], clev=clev)
    if (u is not None) & (v is not None):
        d = 1
        _lat = lat[::d]#,::d]
        _lon = lon[::d]#,::d]
        # uu, vv = rotate_wind(u[::d,::d], v[::d,::d], _lon)
        uu, vv = u[::d,::d].data, v[::d,::d].data
        vec = plot_map.add_vector(ax,
            _lon.data, _lat.data, uu, vv,
            vector_kwargs=dict(
                scale_units="inches", scale=kwargs.get('vector_scale',15),
            ),
            nomap=True,
        )
        rect = mpl.patches.Rectangle(
            (0., 0),  # (x, y)
            0.25,  # width
            0.1,  # height
            facecolor='white',
            edgecolor='gray',
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.quiverkey(vec, 0.10, 0.05, 5, "5",labelpos='E', coordinates='axes')
    
    if (overlay:=kwargs.get('overlay')) is not None:
        plot_map.add_hatches(ax, lon, lat, overlay, clev=[0, 0.05], zorder=10)
        
    # draw urban
    # plot_map.add_contour(ax, XLON, XLAT, xlc.where(xlc==5,0), colors='gray', clev=[4.9,5.0,5.1], linewidths=0.8)
        
    # add text - title, ananote
    if (title:=kwargs.get('title')) is not None: 
        default_title_kwargs = dict(loc='left')
        default_title_kwargs.update(kwargs.get('title_kwargs', {}))
        ax.set_title(title, **default_title_kwargs)

    if kwargs.get('draw_colorbar'):
        plot_map.add_colorbar(cs, ax, clev[1:-1])
        
    return cs, clev 