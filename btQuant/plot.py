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
        fig = px.line(data, x=x, y=y, title=title, color=color,
                      template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'bar':
        fig = px.bar(data, x=x, y=y[0], title=title, color=color,
                     template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'area':
        fig = px.area(data, x=x, y=y, title=title, color=color,
                      template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'scatter':
        fig = px.scatter(data, x=x, y=y[0], title=title, color=color,
                         template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)
        if regression_line:
            fig.update_traces(mode='markers+lines', line=dict(color="red", width=2))

    elif chart_type == 'hist':
        fig = px.histogram(data, x=y[0], title=title, color=color,
                           template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'box':
        fig = px.box(data, y=y[0], x=color if color else None,
                     title=title, template=template,
                     color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'violin':
        fig = px.violin(data, y=y[0], x=color if color else None,
                        box=True, points='all', title=title,
                        template=template, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == 'heatmap':
        if len(y) < 2:
            raise ValueError("Heatmap requires at least two columns for correlation.")
        corr = data[y].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues',
                        title=title, template=template)

    elif chart_type == '3dscatter':
        fig = px.scatter_3d(data, x=x, y=y[0], z=y[1], title=title,
                            template=template, color=color, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == '3dline':
        fig = px.line_3d(data, x=x, y=y[0], z=y[1], title=title,
                         template=template, color=color, color_discrete_sequence=color_discrete_sequence, **kwargs)

    elif chart_type == '3dsurface':
        x_vals = data[x].values
        y_vals = data[y[0]].values
        z_vals = data[y[1]].values

        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.interp(X, y_vals, z_vals)

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])

        fig.update_layout(
            title=title,
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

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    if chart_type in ['3dscatter', '3dline', '3dsurface']:
        fig.update_layout(scene=dict(
            xaxis=dict(showline=True, linewidth=2, linecolor='black'),
            yaxis=dict(showline=True, linewidth=2, linecolor='black'),
            zaxis=dict(showline=True, linewidth=2, linecolor='black')
        ))

    fig.add_annotation(
        text="btQuant by: Boyko Wealth",
        xref="paper", yref="paper",
        x=1, y=-0.12, showarrow=False,
        font=dict(size=12, color="gray"),
        xanchor='right'
    )

    fig.update_traces(
        opacity=0.8,
        hoverinfo='x+y+text'
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or (y[0] if isinstance(y, list) else y),
        title=dict(
            text=title,
            font=dict(size=26, color="black"),
            x=0.01,
            xanchor="left"
        ),
        margin=dict(t=80, b=100),
        font=dict(family="Arial", size=14, color="black"),
    )

    fig.show()