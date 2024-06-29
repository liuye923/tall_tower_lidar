import matplotlib as mpl
def plot_ts(*lines, **kwargs):
    '''
    lines: (x, y, kwargs)
    '''
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.family'] = 'Helvetica'  

    if (fig:=kwargs.get('fig')) is None:  
        fig = plt.figure(figsize=(8, 5))
    if (ax:=kwargs.get('ax')) is None:  ax = plt.axes()
    ax.set_axis_on()
    
    default_line_kw = dict()
    cs = []
    for line in lines:
        x, y, kw = line
        default_line_kw.update(kw)
        _cs, = ax.plot(x, y, **default_line_kw)
        cs.append(cs)
        
    if (title:=kwargs.get('title')) is not None: 
        default_title_kwargs = dict(loc='left')
        default_title_kwargs.update(kwargs.get('title_kwargs', {}))
        ax.set_title(title, **default_title_kwargs)
    if (xlabel:=kwargs.get('xlabel')) is not None: 
        default_xlabel_kwargs = dict()
        default_xlabel_kwargs.update(kwargs.get('xlabel_kwargs', {}))
        ax.set_xlabel(xlabel, **default_xlabel_kwargs)    
    if (ylabel:=kwargs.get('ylabel')) is not None: 
        default_ylabel_kwargs = dict()
        default_ylabel_kwargs.update(kwargs.get('ylabel_kwargs', {}))
        ax.set_ylabel(ylabel, **default_ylabel_kwargs) 
    if (kwargs.get('time_xaxis', False)):
        import matplotlib.dates as mdates
        major_locator = mdates.DayLocator()    
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
    if (kwargs.get('draw_grid', False)):
        default_grid_kwargs = dict(ls='--')
        default_grid_kwargs.update(kwargs.get('grid_kwargs', {}))
        ax.grid(**default_grid_kwargs)
    if (kwargs.get('draw_legend', False)):
        default_legend_kwargs = dict()
        default_legend_kwargs.update(kwargs.get('legend_kwargs', {}))
        ax.legend(**default_legend_kwargs)
    if (ylim:=kwargs.get('ylim')) is not None: ax.set_ylim(ylim)
    if (xlim:=kwargs.get('xlim')) is not None: ax.set_xlim(xlim)
    if (xticks:=kwargs.get('xticks')) is not None: ax.set_xticks(xticks)    
    return ax, cs
