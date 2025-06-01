# visual.py
import pandas as pd
import plotly.express as px

def create_skill_gap_chart(gap_data: dict) -> px.bar:
    """
    Create a Plotly bar chart to visualize the skill gap between a resume and job description.

    Args:
        gap_data (dict): A dictionary with two keys: "Resume" and "Job Description",
                         each mapping to a dictionary of skills and their presence (0 or 1).

    Returns:
        px.bar: A Plotly bar chart figure.
    """
    # Prepare data for the chart
    skills = list(gap_data["Resume"].keys())
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
        title="Skill Gap Analysis",
        labels={"Presence": "Skill Presence (1 = Present, 0 = Absent)"},
        height=400
    )

    # Customize the y-axis to show "Absent" and "Present"
    fig.update_yaxes(
        range=[0, 1],
        tickvals=[0, 1],
        ticktext=["Absent", "Present"],
        title="Skill Presence"
    )

    # Customize the x-axis
    fig.update_xaxes(title="Skills")

    # Update layout for better appearance
    fig.update_layout(
        legend_title_text="Source",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=50, l=50, r=50)
    )

    return fig