import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

boyko_palette = [
    "#000000",  # Black
    "#4B4B4B",  # Dark Grey
    "#888888",  # Medium Grey
    "#1f77b4",  # Blue
    "#2878c2",  # Slightly brighter blue
    "#339af0",  # Light Blue
    "#a6c8ff"   # Pale Blue
]

def plot_chart(
    data: pd.DataFrame,
    chart_type: str,
    x: str = None,
    y: list = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    color: str = None,
    width: int = 1820,
    height: int = 920,
    template: str = 'plotly_white',
    regression_line: bool = False,
    **kwargs
):
    """
    Branded Plotly chart builder with Boyko Wealth styling.

    Parameters:
    - data: pd.DataFrame — Input data.
    - chart_type: str — Type of plot: 'line', 'bar', 'area', 'scatter', 'hist', 'box', 'violin', 'heatmap', '3dscatter', '3dline', '3dsurface', 'regression' etc.
    - x: str — Column for x-axis.
    - y: list of str — Column(s) for y-axis.
    - title: str — Main chart title.
    - xlabel, ylabel: str — Axis labels.
    - color: str — Optional column for grouping by color.
    - width, height: int — Chart dimensions.
    - template: str — Plotly template style.
    - regression_line: bool — Whether to include a regression line (for scatter).
    - kwargs: Additional keyword arguments for the plot function.

    Returns:
    - Plotly Figure object rendered inline.
    """
    
    color_discrete_sequence = boyko_palette

    if chart_type == 'line':
        fig = px.line(data, x=x, y=y, color=color,
                      template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'bar':
        fig = px.bar(data, x=x, y=y[0], color=color,
                     template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'area':
        fig = px.area(data, x=x, y=y, color=color,
                      template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'scatter':
        fig = px.scatter(data, x=x, y=y[0], color=color,
                         template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)
        if regression_line:
            fig.update_traces(mode='markers+lines', line=dict(color="red", width=2))

    elif chart_type == 'hist':
        fig = px.histogram(data, x=y[0], color=color,
                           template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'box':
        fig = px.box(data, y=y[0], x=color if color else None,
                     template=template,
                     color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'violin':
        fig = px.violin(data, y=y[0], x=color if color else None,
                        box=True, points='all',
                        template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'heatmap':
        if len(y) < 2:
            raise ValueError("Heatmap requires at least two columns for correlation.")
        corr = data[y].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues',
                        template=template)

    elif chart_type == '3dscatter':
        fig = px.scatter_3d(data, x=x, y=y[0], z=y[1],
                            template=template, color=color, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == '3dline':
        fig = px.line_3d(data, x=x, y=y[0], z=y[1],
                         template=template, color=color, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == '3dsurface':
        x_vals = data[x].values
        y_vals = data[y[0]].values
        z_vals = data[y[1]].values

        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.interp(X, y_vals, z_vals)

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])

        fig.update_layout(
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title="Z",
            ),
            template=template,
            coloraxis_showscale=False
        )

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    fig.update_xaxes(
        showline=True, 
        linewidth=1.5, 
        linecolor='#333333', 
        mirror=False,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#E0E0E0',
        tickfont=dict(size=12, color='#333333'),
        title_font=dict(size=14, color='#333333')
    )
    
    fig.update_yaxes(
        showline=True, 
        linewidth=1.5, 
        linecolor='#333333', 
        mirror=False,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#E0E0E0',
        tickfont=dict(size=12, color='#333333'),
        title_font=dict(size=14, color='#333333')
    )
    
    if chart_type in ['3dscatter', '3dline', '3dsurface']:
        fig.update_layout(scene=dict(
            xaxis=dict(showline=True, linewidth=1.5, linecolor='#333333'),
            yaxis=dict(showline=True, linewidth=1.5, linecolor='#333333'),
            zaxis=dict(showline=True, linewidth=1.5, linecolor='#333333')
        ))

    if title:
        fig.add_annotation(
            text=title,
            xref="paper", yref="paper",
            x=0, y=-0.05, showarrow=False,
            font=dict(size=11, color="#666666"),
            xanchor='left',
            yanchor='top'
        )

    if chart_type in ['line', 'area']:
        fig.update_traces(
            opacity=0.85,
            hoverinfo='x+y+text',
            line=dict(width=2.5)
        )
    elif chart_type == 'scatter':
        fig.update_traces(
            opacity=0.85,
            hoverinfo='x+y+text',
            marker=dict(size=6, line=dict(width=0.5, color='white'))
        )
    elif chart_type == 'bar':
        fig.update_traces(
            opacity=0.85,
            hoverinfo='x+y+text',
            marker=dict(line=dict(width=0.5, color='white'))
        )
    else:
        fig.update_traces(
            opacity=0.85,
            hoverinfo='x+y+text'
        )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or (y[0] if isinstance(y, list) else y),
        margin=dict(t=20, b=30, l=40, r=20),  # Much smaller margins
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#333333"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True if color else False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ) if color else {}
    )
    
    fig.update_layout(title=None)

    fig.show()