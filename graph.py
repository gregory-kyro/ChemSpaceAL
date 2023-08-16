import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.kaleido.scope.mathjax = None


class Graph:
    def __init__(self, base_path):
        self.base_path = base_path
        self.title_size = 20
        self.axis_title_size = 14
        self.tick_font_size = 12
        self.annotation_size = 14
        self.text_color = "#333333"
        self.background = "white"
        self.grid_color = "#e2e2e2"
        self.line_color = "#000000"
        self.font_family = "Helvetica"
        self.showlegend = True
        self.width = 600
        self.height = 400
        self.title = ""
        self.xaxis_title = ""
        self.yaxis_title = ""
        self.show_xgrid = False
        self.show_ygrid = False
        self.show_yzero = False
        self.show_xzero = False
        self.yrange = None
        self.xrange = None
        self.xgrid_width = 1
        self.ygrid_width = 1
        self.xline_width = 1
        self.yline_width = 1
        self.xmirror = False
        self.ymirror = False
        self.xshowticklabels = True
        self.yshowticklabels = True
        self.xdticks = None
        self.ydticks = None
        self.title_ycoord = 0.95
        self.title_xcoord = 0.5

    def create_folders(self, folder_lists, current_path=None):
        if not folder_lists:
            return
        if current_path is None:
            current_path = self.base_path
        for parent_folder in folder_lists[0]:
            current_path = os.path.join(current_path, parent_folder)
            if not os.path.exists(current_path):
                os.mkdir(current_path)
            create_folders(folder_lists[1:], current_path)

    def update_parameters(self, params):
        for key, val in params.items():
            setattr(self, key, val)

    def style_figure(self, figure):
        layout_dict = dict(
            showlegend=self.showlegend,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor=self.background,
            paper_bgcolor=self.background,
            title=dict(
                text=self.title,
                y=self.title_ycoord,
                x=self.title_xcoord,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=self.title_size,
                    color=self.text_color,
                    family=self.font_family,
                ),
            ),
            height=self.height,  # Set fixed size ratio 3:4
            width=self.width,
            font=dict(
                family=self.font_family,
                size=self.tick_font_size,
                color=self.text_color,
            ),
            legend=dict(
                font=dict(
                    family=self.font_family,
                    size=self.tick_font_size,
                    color=self.text_color,
                ),
            ),
        )
        xaxis_dict = dict(
            title=self.xaxis_title,
            title_font=dict(
                size=self.axis_title_size,
                color=self.text_color,
                family=self.font_family,
            ),
            tickfont=dict(
                size=self.tick_font_size,
                color=self.text_color,
                family=self.font_family,
            ),
            showgrid=self.show_xgrid,
            zeroline=self.show_xzero,
            gridwidth=self.xgrid_width,
            gridcolor=self.grid_color,
            linecolor=self.line_color,  # make x axis line visible
            linewidth=self.xline_width,
            mirror=self.xmirror,
            showticklabels=self.xshowticklabels,
        )

        yaxis_dict = dict(
            title=self.yaxis_title,
            title_standoff=0,
            title_font=dict(
                size=self.axis_title_size,
                color=self.text_color,
                family=self.font_family,
            ),
            tickfont=dict(
                size=self.tick_font_size,
                color=self.text_color,
                family=self.font_family,
            ),
            showgrid=self.show_ygrid,
            zeroline=self.show_yzero,
            gridwidth=self.ygrid_width,
            gridcolor=self.grid_color,
            linecolor=self.line_color,  # make y axis line visible
            linewidth=self.yline_width,
            mirror=self.ymirror,
            showticklabels=self.yshowticklabels,
        )
        if self.xrange is not None:
            xaxis_dict["range"] = self.xrange
        if self.yrange is not None:
            yaxis_dict["range"] = self.yrange
        if self.xdticks is not None:
            xaxis_dict["dtick"] = self.xdticks
        if self.ydticks is not None:
            yaxis_dict["dtick"] = self.ydticks

        figure.update_layout(layout_dict)
        figure.update_xaxes(xaxis_dict)
        figure.update_yaxes(yaxis_dict)
        for annotation in figure["layout"]["annotations"]:
            annotation["font"] = dict(size=self.annotation_size)
        return figure

    def save_figure(self, figure, path, fname, jpg=True, svg=True, pdf=False, html=False):
        if html: figure.write_html(f"{path}html/{fname}.html", include_plotlyjs="cdn")
        if jpg: figure.write_image(f"{path}jpg/{fname}.jpg", scale=4.0,)
        if svg: figure.write_image(f"{path}svg/{fname}.svg")
        if pdf: figure.write_image(f"{path}pdf/{fname}.pdf")


# create_folders(BASE_PATH, [["plots"], ["heatmaps", "ligand_distribution", "cluster_distribution"], ["html", "svg", "jpg"]])