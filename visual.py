# visual.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_skill_gap_chart(gap_data: dict) -> go.Figure:
    """
    Create a polished Plotly bar chart to visualize the skill gap between a resume and job description.

    Args:
        gap_data (dict): A dictionary with two keys: "Resume" and "Job Description",
                         each mapping to a dictionary of skills and their presence (0 or 1).

    Returns:
        go.Figure: A Plotly bar chart figure.
    """
    # Prepare data for the chart
    skills = list(gap_data["Resume"].keys())
    if not skills:
        return go.Figure()  # Return an empty figure if no skills are provided

    resume_values = [gap_data["Resume"][skill] for skill in skills]
    jd_values = [gap_data["Job Description"][skill] for skill in skills]

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        "Skill": skills * 2,
        "Presence": resume_values + jd_values,
        "Source": ["Resume"] * len(skills) + ["Job Description"] * len(skills)
    })

    # Create the bar chart using Plotly Express
    fig = px.bar(
        df,
        x="Skill",
        y="Presence",
        color="Source",
        barmode="group",
        color_discrete_map={"Resume": "#4CAF50", "Job Description": "#2196F3"},
        height=450,
        width=800
    )

    # Customize the y-axis
    fig.update_yaxes(
        range=[-0.1, 1.1],  # Slightly expand the range for better visibility
        tickvals=[0, 1],
        ticktext=["Absent", "Present"],
        title="Skill Presence",
        title_font=dict(size=14, family="Arial"),
        tickfont=dict(size=12, family="Arial"),
        gridcolor="rgba(200, 200, 200, 0.3)",
        zeroline=False
    )

    # Customize the x-axis
    fig.update_xaxes(
        title="Skills",
        title_font=dict(size=14, family="Arial"),
        tickfont=dict(size=12, family="Arial"),
        tickangle=45,  # Rotate labels for better readability
        showgrid=False
    )

    # Update the bars for better appearance
    fig.update_traces(
        width=0.3,  # Adjust bar width for clarity
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),  # Add border to bars
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>",  # Custom hover info
    )

    # Update the layout for a polished look
    fig.update_layout(
        title=dict(
            text="Skill Gap Analysis",
            font=dict(size=18, family="Arial", color="#333"),
            x=0.5,
            xanchor="center"
        ),
        legend_title_text="Source",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, family="Arial")
        ),
        margin=dict(t=100, b=100, l=50, r=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", color="#333"),
        showlegend=True,
        bargap=0.4  # Increase gap between grouped bars
    )

    return fig

def create_skill_gap_heatmap(gap_data: dict) -> go.Figure:
    """
    Create a Plotly heatmap to visualize the skill gap between a resume and job description.

    Args:
        gap_data (dict): A dictionary with two keys: "Resume" and "Job Description",
                         each mapping to a dictionary of skills and their presence (0 or 1).

    Returns:
        go.Figure: A Plotly heatmap figure.
    """
    # Prepare data for the heatmap
    skills = list(gap_data["Resume"].keys())
    if not skills:
        return go.Figure()  # Return an empty figure if no skills are provided

    sources = ["Resume", "Job Description"]
    values = [
        [gap_data["Resume"][skill] for skill in skills],
        [gap_data["Job Description"][skill] for skill in skills]
    ]

    # Create the heatmap using Plotly Graph Objects
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=skills,
        y=sources,
        colorscale=[[0, "#FF6347"], [1, "#32CD32"]],  # Red for absent, green for present
        showscale=True,
        colorbar=dict(
            tickvals=[0, 1],
            ticktext=["Absent", "Present"],
            title="Skill Presence",
            titleside="right",
            titlefont=dict(size=12, family="Arial"),
            tickfont=dict(size=12, family="Arial")
        ),
        hovertemplate="<b>%{y}</b><br>Skill: %{x}<br>Presence: %{z}<extra></extra>"
    ))

    # Update the layout for a polished look
    fig.update_layout(
        title=dict(
            text="Skill Gap Heatmap",
            font=dict(size=18, family="Arial", color="#333"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Skills",
            title_font=dict(size=14, family="Arial"),
            tickfont=dict(size=12, family="Arial"),
            tickangle=45,
            showgrid=False
        ),
        yaxis=dict(
            title="Source",
            title_font=dict(size=14, family="Arial"),
            tickfont=dict(size=12, family="Arial"),
            showgrid=False
        ),
        margin=dict(t=100, b=100, l=50, r=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", color="#333"),
        height=400,
        width=800
    )

    return fig