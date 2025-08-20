import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np


def shorten_title(title, max_length=15):
    return title if len(title) <= max_length else title[:max_length] + "..."

def shorten_response(response, max_words=3):
    if isinstance(response, list):
        return " ".join(response[:max_words])
    return response


def plot_unique_students_per_module(data_map):

    module_counts = {}
    for module_title, df in data_map.items():
        if df is None or df.empty:
            module_counts[module_title] = 0
            continue

        unique_students = df["actor.name"].nunique()
        module_counts[module_title] = unique_students

    module_counts_df = (pd.DataFrame(list(module_counts.items()), columns=["Module Title", "Unique Students"]))

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(
        module_counts_df["Module Title"],
        module_counts_df["Unique Students"],
        color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title("Unique Students per Module", fontsize=14)
    ax.set_xlabel("Module Title", fontsize=12)
    ax.set_ylabel("Unique Students", fontsize=12)
    ax.set_xticklabels(module_counts_df["Module Title"], rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    return fig


def average_completion_time(df, outlier_threshold=100):
    if "slide_number" in df.columns:
        slides_ordered = (
            df[["slide_number","object.definition.name.und"]]
            .drop_duplicates()
            .sort_values("slide_number")["object.definition.name.und"]
            .tolist())
    else:
        slides_ordered = sorted(df["object.definition.name.und"].unique())

    avg_prog, std_prog = calculate_avg_progression(df, slides_ordered, outlier_threshold)
    if not avg_prog:
        return 0.0
    return float(avg_prog[-1])



def calculate_avg_active_time(df, gap_threshold=10):

    df["timestamp_gmt8"] = pd.to_datetime(df["timestamp_gmt8"], format="ISO8601")

    df = df.sort_values(by=["actor.name", "timestamp_gmt8"])

    df["elapsed_minutes"] = (df["timestamp_gmt8"] - df.groupby("actor.name")["timestamp_gmt8"].transform("min")).dt.total_seconds() / 60
    df["time_gap"] = df.groupby("actor.name")["elapsed_minutes"].diff().fillna(0)
    df["is_large_gap"] = df["time_gap"] > gap_threshold

    adjusted_elapsed_times = []
    for user, user_data in df.groupby("actor.name"):
        last_elapsed_time = 0
        user_adjusted_times = []
        for _, row in user_data.iterrows():
            if row["is_large_gap"]:
                pass
            else:
                last_elapsed_time += row["time_gap"]
            user_adjusted_times.append(last_elapsed_time)
        adjusted_elapsed_times.extend(user_adjusted_times)

    df["adjusted_elapsed_time"] = adjusted_elapsed_times
    df["active_time"] = df.groupby("actor.name")["adjusted_elapsed_time"].diff().fillna(0)
    df["active_time"] = df["active_time"].clip(lower=0)

    avg_active_time = df.groupby("object.definition.name.und")["active_time"].mean().reset_index()

    return avg_active_time


def average_completion_time_per_module(df, gap_threshold=100):
    """
    For each user: (last_timestamp - first_timestamp).
    Then exclude any user who exceeds outlier_threshold, 
    and average the rest.
    """
    import pandas as pd
    import numpy as np

    if df is None or df.empty:
        return 0.0

    # Convert
    df["timestamp_gmt8"] = pd.to_datetime(df["timestamp_gmt8"], format="ISO8601")
    # Sort by user and timestamp
    df = df.sort_values(["actor.name", "timestamp_gmt8"])

    # Group by user to find first and last
    grouped = df.groupby("actor.name")["timestamp_gmt8"]
    user_first = grouped.transform("min")
    user_last = grouped.transform("max")

    # For each row, compute total time in minutes for that user's entire progression
    df["user_total_time"] = (user_last - user_first).dt.total_seconds() / 60

    # If you want to exclude any user with total_time > outlier_threshold
    # We can do it by user or row, e.g.:
    df_filtered = df[df["user_total_time"] <= gap_threshold].copy()
    if df_filtered.empty:
        return 0.0

    # The final average is the mean of user_total_time, but per user, so:
    # group by user, then take the first row from each user
    user_level = (
        df_filtered.groupby("actor.name")["user_total_time"].first()
    )
    # The average across users
    return user_level.mean()




def plot_avg_time_per_module(data_map, gap_threshold=100):
    """
    For each module in data_map, compute average first→last event time,
    then bar plot the results in the dictionary order.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    module_names = list(data_map.keys())  # preserve dictionary order
    module_times = []
    for module_title in module_names:
        df = data_map[module_title]
        avg_time = average_completion_time_per_module(df, gap_threshold)
        module_times.append(avg_time)

    result_df = pd.DataFrame({
        "Module Title": module_names,
        "Average Time (mins)": module_times
    })

    fig, ax = plt.subplots(figsize=(15,7))
    ax.bar(
        result_df["Module Title"],
        result_df["Average Time (mins)"],
        color="skyblue", edgecolor="black", alpha=0.7
    )
    ax.set_title("Average Completion Time per Module", fontsize=14)
    ax.set_xlabel("Module Title", fontsize=12)
    ax.set_ylabel("Time (minutes)", fontsize=12)
    ax.set_xticklabels(result_df["Module Title"], rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig



def plot_day_attempted(df, data_key):

    df['timestamp_gmt8'] = pd.to_datetime(df['timestamp_gmt8'], format='ISO8601')
    df = df.sort_values(by=["actor.name", "timestamp_gmt8"])

    attempted_df = df[df["verb.display.en-US"] == "attempted"]

    first_attempt_df = attempted_df.groupby("actor.name").first().reset_index()
    first_attempt_df["day_of_week"] = first_attempt_df["timestamp_gmt8"].dt.day_name()

    day_count_map = first_attempt_df["day_of_week"].value_counts()
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_count_map = day_count_map.reindex(ordered_days, fill_value=0)

    total_students = len(first_attempt_df)
    day_percentages = {
        day: (count / total_students) * 100 if total_students > 0 else 0
        for day, count in day_count_map.items()}

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(day_count_map.index, day_count_map.values, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title(f"Day of First Attempt | ({data_key})", fontsize=14)
    ax.set_xlabel("Week Days", fontsize=12)
    ax.set_ylabel("Frequency of Students", fontsize=12)
    ax.set_xticklabels(day_count_map.index, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()

    days = list(day_count_map.index)
    counts = list(day_count_map.values)
    percentages = [day_percentages[d] for d in day_count_map.index]

    day_df = pd.DataFrame({
        "Day": days,
        "Count": counts,
        "Percentage (%)": [f"{p:.1f}" for p in percentages]})

    day_df.loc[len(day_df)] = ["TOTAL", total_students, "100.0"]

    return fig, day_df



def plot_mean_break_time_with_yzoom(df, data_key, y_max):

    df = df.sort_values(by="timestamp_gmt8")

    break_times_per_slide = []

    for person, group in df.groupby("actor.name"):
        group["elapsed_minutes"] = ((group["timestamp_gmt8"] - group["timestamp_gmt8"].min()).dt.total_seconds() / 60)
        group["time_gap"] = group["elapsed_minutes"].diff().fillna(0)

        group["is_large_gap"] = group["time_gap"] > 10

        break_data = group[group["is_large_gap"]]
        for _, row in break_data.iterrows():
            break_times_per_slide.append({
                "slide_title": row["object.definition.name.und"],
                "break_time": row["time_gap"]})

    break_times_df = pd.DataFrame(break_times_per_slide)

    if break_times_df.empty:
        fig, ax = plt.subplots(figsize=(6,4), facecolor = "#ecf0f1")
        ax.set_title(f"No Break Times Found ({data_key})")
        return fig

    mean_break_times = break_times_df.groupby("slide_title")["break_time"].mean().reset_index()
    mean_break_times = mean_break_times.sort_values(by="break_time", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor = "#ecf0f1")
    ax.bar(
        mean_break_times["slide_title"].apply(lambda x: shorten_title(x)),
        mean_break_times["break_time"],
        color="skyblue",
        edgecolor="black",
        alpha=0.7)
    ax.set_xlabel("Slide Title", fontsize=12)
    ax.set_ylabel("Average Break Time (minutes)", fontsize=12)
    ax.set_title(f"Average Break Time per Slide ({data_key})", fontsize=14)
    ax.set_xticklabels(mean_break_times["slide_title"].apply(lambda x: shorten_title(x)), rotation=45, ha="right")
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_facecolor("#ecf0f1")
    fig.tight_layout()
    return fig


def plot_mean_time_per_slide(df, data_key, gap_threshold):

    df["timestamp_gmt8"] = pd.to_datetime(df["timestamp_gmt8"], format="ISO8601")

    df = df.sort_values(by=['actor.name', 'timestamp_gmt8'])

    df['next_timestamp'] = df.groupby('actor.name')['timestamp_gmt8'].shift(-1)
    df['time_spent_minutes'] = ((df['next_timestamp'] - df['timestamp_gmt8']).dt.total_seconds() / 60)

    df['is_large_gap'] = df['time_spent_minutes'] > gap_threshold
    df.loc[df['is_large_gap'], 'time_spent_minutes'] = gap_threshold

    df['time_spent_minutes'] = df['time_spent_minutes'].fillna(0)

    mean_time_per_slide = df.groupby(['slide_number', 'object.definition.name.und']).agg(mean_time_per_person=('time_spent_minutes', 'mean')).reset_index()
    mean_time_per_slide = mean_time_per_slide.sort_values(by='slide_number')

    fig, ax = plt.subplots(figsize=(17, 8))
    ax.bar(
        mean_time_per_slide['object.definition.name.und'].apply(lambda x: shorten_title(x)),
        mean_time_per_slide['mean_time_per_person'],
        color="skyblue",
        edgecolor="black",
        alpha=0.7)
    ax.set_title(f'Average Engagement Time Spent on Each Slide ({data_key})')
    ax.set_xlabel('Slide Title (Chronological Order)')
    ax.set_ylabel('Average Time Spent (minutes)')
    ax.set_xticklabels(mean_time_per_slide['object.definition.name.und'].apply(lambda x: shorten_title(x)), rotation=45, ha='right')
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    return fig


def plot_count_per_slide(df, data_key):
    
    student_counts_per_slide = df.groupby(['slide_number', 'object.definition.name.und']).agg(
        unique_students=('actor.name', 'nunique')).reset_index()

    student_counts_per_slide = student_counts_per_slide.sort_values(by='slide_number')

    student_counts_per_slide['unique_title'] = (
        student_counts_per_slide['object.definition.name.und'] + 
        " (Slide " + 
        student_counts_per_slide['slide_number'].astype(str) + 
        ")")

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(
        student_counts_per_slide['object.definition.name.und'],
        student_counts_per_slide['unique_students'],
        marker='o',
        linestyle='-',
        color = 'blue')
    
    ax.set_title(f'Unique Student Count by Slide Title in Dataset ({data_key})')
    ax.set_xlabel('Slide Title (Chronological Order)')
    ax.set_ylabel('Number of Students')
    ax.axhline(y=student_counts_per_slide["unique_students"].max(), color="blue", linestyle="--", linewidth=1)
    ax.set_xticklabels(student_counts_per_slide['object.definition.name.und'].apply(lambda x: shorten_title(x)), rotation=45, ha='right')
    ax.grid(True)
    fig.tight_layout()

    return fig



# In plots.py
# 1) Keep your existing function EXACTLY as-is:
def calculate_avg_progression(df, slide_titles, outlier_threshold=100):
    """
    Compute average progression ignoring users who exceed outlier_threshold minutes overall.
    Also returns a std array if you want ±1 stdev shading.
    
    EXACT logic from your original code:
    - Sort by user/timestamp
    - Convert to elapsed_minutes
    - Filter out users/timepoints > outlier_threshold
    - For each slide in slide_titles, gather sums of time_increment for that subset across all users
    - Return (avg_progression, std_progression)
    """
    df = df.sort_values(by=["actor.name", "timestamp_gmt8"])

    # Elapsed time
    df["elapsed_minutes"] = (
        df["timestamp_gmt8"] 
        - df.groupby("actor.name")["timestamp_gmt8"].transform("min")
    ).dt.total_seconds() / 60

    # Filter out rows > outlier_threshold
    df = df[df["elapsed_minutes"] <= outlier_threshold]

    df["time_increment"] = (
        df.groupby(["actor.name"])["elapsed_minutes"].diff().fillna(0)
    )

    avg_progression = []
    std_progression = []

    for slide in slide_titles:
        slide_data = df[df["object.definition.name.und"] == slide]

        user_cumulative_times = []
        if not slide_data.empty:
            for user in df["actor.name"].unique():
                user_data = df[
                    (df["actor.name"] == user)
                    & (df["object.definition.name.und"].isin(
                        slide_titles[: slide_titles.index(slide) + 1]
                      ))
                ]
                if not user_data.empty:
                    user_cumulative_times.append(user_data["time_increment"].sum())

        if user_cumulative_times:
            mean_val = np.mean(user_cumulative_times)
            std_val = np.std(user_cumulative_times)
        else:
            mean_val = 0
            std_val = 0

        avg_progression.append(mean_val)
        std_progression.append(std_val)

    return avg_progression


def plot_individual_completion(df, data_key, person):
    df['timestamp_gmt8'] = pd.to_datetime(df['timestamp_gmt8'], format='ISO8601')

    gap_threshold = 10

    # Filter by specific person selected
    person_data = df[df["actor.name"] == person]

    # In case user tries to access a person outside the dataset
    if person_data.empty:
        print(f"No data available for {person} in {data_key}.")
        return

    # Sort by timestamp
    person_data = person_data.sort_values(by="timestamp_gmt8")

    # Define slide count map
    slide_count = {}

    # Define slide count dataset
    slide_count_data = person_data.groupby(["object.definition.name.und"])

    # Count the occurrences of the slide and store in the map
    for slide_title, slide_group in slide_count_data:
        if len(slide_group) > 1:
            slide_count[slide_title] = len(slide_group)

    # Grouping by slide title, taking the first occurrence
    progression_data = person_data.groupby(["object.definition.name.und"]).first().reset_index()

    # Ensure the x-axis is dynamic by sorting by time
    progression_data = progression_data.sort_values(by="timestamp_gmt8").reset_index(drop=True)

    # Create elapsed time in minutes
    progression_data["elapsed_minutes"] = ((progression_data["timestamp_gmt8"] - progression_data["timestamp_gmt8"].min()).dt.total_seconds() / 60)

    # Identify large times based on threshold, filling NA's with 0
    progression_data["time_gap"] = progression_data["elapsed_minutes"].diff().fillna(0)
    progression_data["is_large_gap"] = progression_data["time_gap"] > gap_threshold

    adjusted_elapsed_times = []
    last_elapsed_time = 0
    break_times = []
    # Store break times for legend
    for i, row in progression_data.iterrows():
        if row["is_large_gap"]:
            break_times.append(int(row["time_gap"]))
        else:
            last_elapsed_time += row["time_gap"]
        adjusted_elapsed_times.append(last_elapsed_time)

    progression_data["adjusted_elapsed_minutes"] = adjusted_elapsed_times

    # Pull unique slide titles in the order completed for the x-axis
    slide_titles = progression_data["object.definition.name.und"].tolist()
    short_titles = progression_data["object.definition.name.und"].apply(lambda x: shorten_title(x)).tolist()

    # Dynamic y-axis based on the persons max elapsed time + buffer time
    y_max = max(progression_data["adjusted_elapsed_minutes"].max() + 10, 30)

    # Plot main progression points
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.plot(
        slide_titles,
        progression_data["adjusted_elapsed_minutes"],
        marker="o",
        color="blue",
        label=person)

    ax.set_xlabel("Slide Title (Completed Order)", fontsize=12)
    ax.set_ylabel("Total Elapsed Time After Breaks (minutes)", fontsize=12)
    ax.set_title(f"Progression Through Slides by {person} ({data_key})", fontsize=14)
    ax.set_xticklabels(short_titles, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, y_max)
    ax.axhline(y=20, color="red", linestyle="--", linewidth=1, label="Max Expected Time")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Assign unique colors to each break, and create map for colors used before
    break_colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33FF", "#57FFFF", "#FFA500", "#8B0000", "#FFD700", "#4B0082", "#FF69B4"]
    used_colors = {}

    # Add break points on graph by assigning a color to them
    for i, row in progression_data.iterrows():
        if row["is_large_gap"]:
            break_time = int(row["time_gap"])
            if break_time not in used_colors:
                color_index = len(used_colors) % len(break_colors)
                used_colors[break_time] = break_colors[color_index]

            # Plotting breakpoints, with larger sizes to ensure readability
            ax.plot(
                i - 1,
                row["adjusted_elapsed_minutes"],
                marker="o",
                color=used_colors[break_time],
                markersize=10,
                label=f"Break: {break_time} mins" if f"Break: {break_time} mins" not in used_colors else None)

    # Add breaks to the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Legend")
    fig.tight_layout()
    return fig, slide_count