class Graph:
    def __init__(self):
        self.title_size = 20
        self.axis_title_size = 14
        self.tick_font_size = 12
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

    def update_parameters(self, params):
        for key, val in params.items():
            setattr(self, key, val)

    def style_figure(figure):
        figure.update_layout(
            dict(
                showlegend=self.showlegend,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor=self.background,
                paper_bgcolor=self.background,
                title=dict(
                    text=self.title,
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
        )

        # Setting the title size and color and grid for both x and y axes
        figure.update_xaxes(
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
            showgrid=True,
            gridwidth=1,
            gridcolor=self.grid_color,
            linecolor=self.line_color,  # make x axis line visible
            linewidth=2,
        )

        figure.update_yaxes(
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
            showgrid=True,
            gridwidth=1,
            gridcolor=self.grid_color,
            linecolor=self.line_color,  # make y axis line visible
            linewidth=2,
        )
        return figure
