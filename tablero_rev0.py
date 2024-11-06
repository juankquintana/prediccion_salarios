import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.colors

# Incorporate data
#url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/refs/heads/main/Data/data_top10.csv'
url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/refs/heads/branch_alejandra/Data/data_top10_country.csv'
df= pd.read_csv(url)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Get unique positions for the dropdown
positions = df['job_title'].unique()

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Comparación salario promedio por país y continente"), width=6),
        dbc.Col(html.H5("Posición:"), width="auto"),
        dbc.Col(
            dcc.Dropdown(
                id='position-dropdown',
                options=[{'label': pos, 'value': pos} for pos in positions],
                value=positions[0],
                clearable=False
            ),
            width=2
        )
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='map-fig', style={'height': '400px'}), width=8)
    ], justify="left"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='bar-fig', style={'height': '400px'}), width=8)
    ], justify="left") 
], fluid=True)

# Callback to update figures based on the selected position
@app.callback(
    [Output('map-fig', 'figure'),
     Output('bar-fig', 'figure')],
    [Input('position-dropdown', 'value')]
)
def update_figures(selected_position):
    # Filter data for the selected position
    filtered_df = df[df['job_title'] == selected_position]
    
    # Calculate the median salary by country for the chosen position
    country_median_df = filtered_df.groupby("employee_country", as_index=False).median(numeric_only=True)
    
    # Create the updated map figure
    map_fig = px.choropleth(
        country_median_df,
        locations="employee_country",
        locationmode="country names",
        color="salary_in_usd",
        hover_name="employee_country",
        color_continuous_scale="Viridis",
        title=f"Salario promedio por país - {selected_position}",
    )

    # Adjust layout for larger display and use a different map style
    map_fig.update_geos(
        projection_type="robinson",       # Robinson projection to display continents accurately
        showcoastlines=True,              # Show coastlines
        coastlinecolor="Black",           # Color of coastlines
        showland=True,                    # Show land areas
        landcolor="lightgray",            # Color of land areas
        showocean=True,                   # Show ocean
        oceancolor="lightblue",           # Color of ocean
        showcountries=True,               # Show country borders
        countrycolor="Black"              # Color of country borders
    )
    map_fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},  # Remove excess whitespace
    )

    # Calculate the median salary by region for the chosen position
    region_median_df = filtered_df.groupby("employee_residence", as_index=False).median(numeric_only=True)
    
    # Create the updated bar figure
    bar_fig = px.bar(
        region_median_df,
        x="employee_residence",
        y="salary_in_usd",
        text="salary_in_usd",
        title=f"Salario promedio por continente - {selected_position}"
    )

    # Set colors for bar chart to match `Viridis`
    color_scale = plotly.colors.sequential.Viridis
    min_salary = region_median_df['salary_in_usd'].min()
    max_salary = region_median_df['salary_in_usd'].max()
    region_median_df['color'] = region_median_df['salary_in_usd'].apply(
        lambda x: color_scale[int((x - min_salary) / (max_salary - min_salary) * (len(color_scale) - 1))]
    )
    bar_fig.update_traces(marker_color=region_median_df['color'])
    bar_fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    
    return map_fig, bar_fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)