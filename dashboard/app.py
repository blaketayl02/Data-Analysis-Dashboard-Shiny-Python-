import seaborn as sns
from faicons import icon_svg
from shiny import App, reactive, render, ui

import data_loader
import plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


data_map = data_loader.load_and_clean_data()

#this is precomputed time for average slide time / average module time at the top of the page. This was done to save runtime, but could be removed or implemented differently.
time_lookup_df24 = pd.read_csv("Data test/PSYT/Everything/FinalDatasets/Parsed/Cleaned/UpdatedYears/2024/completion_slide_time_full24.csv")
time_lookup_df23 = pd.read_csv("Data test/PSYT/Everything/FinalDatasets/Parsed/Cleaned/UpdatedYears/2023/completion_slide_time_full.csv")


#computes message stats table for users (maily related to checking costs)
def compute_message_stats(df):
    if df.empty:
        return pd.DataFrame()
    
    results = []
    
    #only focus on rows with both ROLE and CONTENT
    df = df.dropna(subset=["ROLE", "CONTENT", "actor.name"])
    
    roles_of_interest = ["patient", "user", "user-level2", "preceptor", "preceptor-level2"]
    all_users = df["actor.name"].unique()
    
    #preceptor cost calculation (not really relevant)
    preceptor_df = df[df["ROLE"] == "preceptor"]
    preceptor_costs = {}
    
    if not preceptor_df.empty:
        preceptor_df["word_count"] = preceptor_df["CONTENT"].apply(lambda x: len(str(x).split()))
        preceptor_df["message_cost"] = np.ceil(preceptor_df["word_count"] / 30)
        
        #sum costs for all students
        for user in all_users:
            user_preceptor_df = preceptor_df[preceptor_df["actor.name"] == user]
            if not user_preceptor_df.empty:
                preceptor_costs[user] = user_preceptor_df["message_cost"].sum()
    
    for role in roles_of_interest:
        role_df = df[df["ROLE"] == role]
        
        if role == "preceptor" and preceptor_costs:
            #cost calculation for preceptor
            cost_values = list(preceptor_costs.values())
            avg_cost = np.mean(cost_values) if cost_values else 0
            max_cost = np.max(cost_values) if cost_values else 0
            min_cost = np.min(cost_values) if cost_values else 0
            
            results.append({
                "Role": role,
                "Avg Messages Per User": 0,
                "Max Messages Per User": 0,
                "Min Messages Per User": 0,
                "Avg Words Per Message": 0,
                "Max Words Per Message": 0,
                "Min Words Per Message": 0,
                "Avg Cost Per User": round(avg_cost, 1),
                "Max Cost Per User": round(max_cost, 1),
                "Min Cost Per User": round(min_cost, 1)
            })
            continue
        
        if role_df.empty:
            results.append({
                "Role": role,
                "Avg Messages Per User": 0,
                "Max Messages Per User": 0,
                "Min Messages Per User": 0,
                "Avg Words Per Message": 0,
                "Max Words Per Message": 0,
                "Min Words Per Message": 0,
                "Avg Cost Per User": 0,
                "Max Cost Per User": 0,
                "Min Cost Per User": 0
            })
            continue
        
        #per user message counts
        user_msg_counts = {}
        for user in all_users:
            user_role_df = role_df[role_df["actor.name"] == user]
            count = len(user_role_df)
            if count > 0:
                user_msg_counts[user] = count
        
        if user_msg_counts:
            msg_counts = list(user_msg_counts.values())
            avg_msg_count = np.mean(msg_counts)
            max_msg_count = np.max(msg_counts)
            min_msg_count = np.min(msg_counts)
        else:
            avg_msg_count = max_msg_count = min_msg_count = 0
        
        role_df["word_count"] = role_df["CONTENT"].apply(lambda x: len(str(x).split()))
        
        if not role_df.empty:
            avg_word_count = role_df["word_count"].mean()
            max_word_count = role_df["word_count"].max()
            min_word_count = role_df["word_count"].min()
        else:
            avg_word_count = max_word_count = min_word_count = 0
        
        results.append({
            "Role": role,
            "Avg Messages Per User": round(avg_msg_count, 1),
            "Max Messages Per User": int(max_msg_count),
            "Min Messages Per User": int(min_msg_count),
            "Avg Words Per Message": round(avg_word_count, 1),
            "Max Words Per Message": int(max_word_count),
            "Min Words Per Message": int(min_word_count),
            "Avg Cost Per User": 0,
            "Max Cost Per User": 0,
            "Min Cost Per User": 0
        })
    
    return pd.DataFrame(results)


#computes per user chatbot time and displays it within a table (including module time in conversation view)
def compute_per_user_chatbot_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["actor.name", "Chatbot Time (mins)", "User Message Count", "Level-2 Message Count", "Module Time (mins)"])

    df = df.copy()
    df["timestamp_gmt8"] = pd.to_datetime(df["timestamp_gmt8"], errors="coerce")
    df.sort_values(["actor.name", "timestamp_gmt8"], inplace=True)

    #only keep rows where ROLE is not NaN for chatbot calculations
    role_df = df.dropna(subset=["ROLE"])

    user_durations = []
    all_users = df["actor.name"].dropna().unique()
    
    #gap threshold to remove breaktimes from total calculation (i.e if they took >10 minute break then exclude 10 minutes from their total time)
    gap_threshold = 10

    for user in all_users:
        person_data = df[df["actor.name"] == user].sort_values(by="timestamp_gmt8")
        
        if person_data.empty:
            user_durations.append({
                "actor.name": user,
                "Chatbot Time (mins)": 0.0,
                "User Message Count": 0,
                "Level-2 Message Count": 0,
                "Module Time (mins)": 0.0,
            })
            continue

        progression_data = person_data.groupby(["object.definition.name.und"]).first().reset_index()
        
        progression_data = progression_data.sort_values(by="timestamp_gmt8").reset_index(drop=True)
        
        if len(progression_data) >= 1:
            progression_data["elapsed_minutes"] = (
                (progression_data["timestamp_gmt8"] - progression_data["timestamp_gmt8"].min()).dt.total_seconds() / 60
            )
            
            #identify large times based on threshold, filling NA's with 0
            progression_data["time_gap"] = progression_data["elapsed_minutes"].diff().fillna(0)
            progression_data["is_large_gap"] = progression_data["time_gap"] > gap_threshold
            
            adjusted_elapsed_times = []
            last_elapsed_time = 0
            
            for i, row in progression_data.iterrows():
                if row["is_large_gap"]:
                    pass
                else:
                    last_elapsed_time += row["time_gap"]
                adjusted_elapsed_times.append(last_elapsed_time)
            
            module_time = adjusted_elapsed_times[-1] if adjusted_elapsed_times else 0.0
        else:
            module_time = 0.0
        
        user_sub = role_df[role_df["actor.name"] == user]
        if user_sub.empty:
            chatbot_time = 0.0
            message_count = 0
            level2_message_count = 0
        else:
            timestamps = sorted(user_sub["timestamp_gmt8"])
            total_elapsed_minutes = 0.0
            for i in range(len(timestamps) - 1):
                time_diff = (timestamps[i+1] - timestamps[i]).total_seconds() / 60.0
                if time_diff <= 20.0:
                    total_elapsed_minutes += time_diff
            
            chatbot_time = total_elapsed_minutes
            message_count = user_sub["ROLE"].value_counts().get('user', 0)
            level2_message_count = user_sub["ROLE"].value_counts().get('user-level2', 0)
        
        user_durations.append({
            "User": user,
            "Chatbot Time (mins)": round(chatbot_time, 3),
            "User Msg Count": message_count,
            "Level-2 Msg Count": level2_message_count,
            "Module Time (mins)": round(module_time, 3),
        })

    table_df = pd.DataFrame(user_durations)
    table_df.sort_values(by="Module Time (mins)", ascending=False, inplace=True)

    return table_df


#computes count of messages across roles
def compute_total_message_counts(df):
    if df.empty:
        return pd.DataFrame()
    
    df = df.dropna(subset=["ROLE", "CONTENT"])
    
    roles_of_interest = ['preceptor', 'preceptor-feedback', 'preceptor-level2', 'preceptor-level2-feedback', 'patient', 'user', 'user-level2', 'user-ddx-summary']
    
    results = []
    total_messages = 0
    
    for role in roles_of_interest:
        count = len(df[df["ROLE"] == role])
        total_messages += count
        results.append({
            "Role": role,
            "Total Messages": count
        })
    
    for result in results:
        if total_messages > 0:
            result["Percentage"] = f"{(result['Total Messages']/total_messages*100):.1f}%"
        else:
            result["Percentage"] = "0.0%"
    
    results.append({
        "Role": "TOTAL",
        "Total Messages": total_messages,
        "Percentage": "100.0%"
    })
    
    return pd.DataFrame(results)


#computes temporal stats based on user message sessions
def compute_temporal_stats(df):
    if df.empty:
        return pd.DataFrame()
    
    user_df = df[df["ROLE"] == "user"].copy()
    
    if user_df.empty:
        return pd.DataFrame()
    
    user_df["timestamp_gmt8"] = pd.to_datetime(user_df["timestamp_gmt8"], errors="coerce")
    
    session_starts = user_df.groupby("actor.name")["timestamp_gmt8"].min().reset_index()
    session_starts.columns = ["user", "session_start"]
    
    session_starts["hour"] = session_starts["session_start"].dt.hour
    session_starts["day_of_week"] = session_starts["session_start"].dt.day_name()
    session_starts["month"] = session_starts["session_start"].dt.to_period('M')
    session_starts["date"] = session_starts["session_start"].dt.date
    
    def categorize_time(hour):
        return hour
    
    session_starts["time_period"] = session_starts["hour"].apply(categorize_time)
    
    stats = []
    
    time_counts = session_starts["time_period"].value_counts()
    for period, count in time_counts.items():
        stats.append({
            "Category": "Time of Day",
            "Subcategory": period,
            "Count": count,
            "Percentage": f"{(count/len(session_starts)*100):.1f}%"})
    
    day_counts = session_starts["day_of_week"].value_counts()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_order:
        if day in day_counts:
            count = day_counts[day]
            stats.append({
                "Category": "Day of Week",
                "Subcategory": day,
                "Count": count,
                "Percentage": f"{(count/len(session_starts)*100):.1f}%"})
    
    month_counts = session_starts["month"].value_counts().sort_index()
    for month, count in month_counts.items():
        stats.append({
            "Category": "Sessions per Month",
            "Subcategory": str(month),
            "Count": count,
            "Percentage": f"{(count/len(session_starts)*100):.1f}%"})
    
    date_counts = session_starts["date"].value_counts().sort_values(ascending=False)
    concurrent_stats = []
    for sessions_count in sorted(date_counts.unique(), reverse=True):
        days_with_count = (date_counts == sessions_count).sum()
        concurrent_stats.append({
            "Category": "Concurrent Sessions",
            "Subcategory": f"{sessions_count} session{'s' if sessions_count != 1 else ''} per day",
            "Count": days_with_count,
            "Percentage": f"{(days_with_count/len(date_counts)*100):.1f}%"})
    
    stats.extend(concurrent_stats)
    
    return pd.DataFrame(stats)


#computes weekly users
def compute_weekly_users(df):
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df["timestamp_gmt8"] = pd.to_datetime(df["timestamp_gmt8"], errors="coerce")
    
    start_date = pd.Timestamp("2025-02-11")
    
    weeks = []
    week_labels = []
    current_date = start_date
    
    for i in range(8):
        end_date = current_date + pd.Timedelta(days=6)
        weeks.append((current_date, end_date))
        week_labels.append(f"Week {i+1}: {current_date.strftime('%b %d')} - {end_date.strftime('%b %d')}")
        current_date = end_date + pd.Timedelta(days=1)
    
    results = []
    
    for i, (week_start, week_end) in enumerate(weeks):
        week_data = df[(df["timestamp_gmt8"] >= week_start) & (df["timestamp_gmt8"] <= week_end)]
        
        if week_data.empty:
            results.append({
                "Week": week_labels[i],
                "Total Unique Users": 0,
                "Users Interacted with Chatbot": 0})
            continue
        
        total_unique = week_data["actor.name"].nunique()
        
        users_with_user_role = week_data[week_data["ROLE"] == "user"]["actor.name"].nunique()
        
        results.append({
            "Week": week_labels[i],
            "Total Unique Users": total_unique,
            "Users Interacted with Chatbot": users_with_user_role})
    
    return pd.DataFrame(results)



#all the UI elements
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "dataset_choice",
            "Choose Dataset:",
            ["Old Kato", "New Kato", "New Kato July"],
            selected="Old Kato"),

        ui.input_select(
            "question",
            "Pick an analysis:",
            [
                "Day Attempted Plot",
                "Average Active Time per Module",
                "Average Active Time per Slide",
                "Unique Student Count per Slide",
                "Student Progression Graph",
                "Conversation View",
            ],
            selected="Day Attempted Plot"
        ),
        ui.output_ui("gap_threshold_ui"),
        ui.output_ui("person_ui"),
        title="Filter controls"
    ),

    ui.panel_title(
        ui.tags.div(
            ui.tags.h1("Mr. Kato Module Analysis", class_="m-2"),
            ui.tags.img(src="ubc.png", height="100px"),
            class_="d-flex justify-content-between align-items-center w-100"
        )
    ),
    ui.layout_column_wrap(
        ui.value_box(
            "Total Unique Students",
            ui.output_text("total_students"),
            showcase=icon_svg("user")
        ),
        ui.value_box(
            ui.output_text("avg_slide_time_label"),
            ui.output_text("avg_slide_time_value"),
            showcase=icon_svg("clock")
        ),
        ui.value_box(
            "Average Completion Time",
            ui.output_text("avg_completion_time"),
            showcase=icon_svg("hourglass-end")
        ),
        fill=False
    ),
    ui.output_ui("main_layout"),
    fillable=True)


#server logic all goes here
def server(input, output, session):
    day_data_value = reactive.Value(None)
    slide_count_value = reactive.Value(None)
    weekly_users_value = reactive.Value(None)

    #person drop down if we are in conversation view
    @output
    @render.ui
    def person_ui():
        if input.question() == "Conversation View":
            df = current_df()
            if df.empty:
                return None

            df = df.dropna(subset=["ROLE","CONTENT","actor.name"])

            users = df["actor.name"].dropna().unique().tolist()
            if not users:
                return ui.HTML("No valid users found.")

            return ui.input_select(
                "person",
                "Select a user:",
                users,
                selected=users[0])
        
        elif input.question() == "Student Progression Graph":
            df = current_df()
            if df.empty:
                return None

            users = df["actor.name"].dropna().unique().tolist()
            if not users:
                return ui.HTML("No valid users found.")

            return ui.input_select(
                "person",
                "Select a user:",
                users,
                selected=users[0])
        
        else:
            return None


    #gets current dataset
    @reactive.Calc
    def current_df():
        dataset_choice = input.dataset_choice()

        if dataset_choice not in data_map:
            return pd.DataFrame()

        return data_map[dataset_choice]
    
    #UI showing gap threshold 
    @output
    @render.ui
    def gap_threshold_ui():
        q = input.question()
        if q == "Average Active Time per Module":
            return ui.input_slider(
                "gap_threshold",
                "Gap Threshold (minutes):",
                min=0,
                max=1000,
                value=30)
        
        elif q == "Average Active Time per Slide":
            return ui.input_slider(
                "gap_threshold",
                "Gap Threshold (minutes):",
                min=10,
                max=500,
                value=10)
        
        else:
            return None
        

    #displays temporal stats table
    @output
    @render.data_frame
    def temporal_stats_table():
        if input.question() != "Conversation View":
            return pd.DataFrame()

        df = current_df()
        if df.empty:
            return pd.DataFrame()

        temporal_df = compute_temporal_stats(df)
        return temporal_df

    #main layout for each dropdown menu
    @output
    @render.ui
    def main_layout():
        q = input.question()
        if q == "Conversation View":
            return ui.layout_columns(
                ui.card(
                    ui.card_header("Conversation"),
                    ui.output_ui("conversation_output"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Conversation Stats"),
                    ui.output_data_frame("conv_stats_table"),
                    ui.output_data_frame("conv_avg_time_table"),
                    ui.output_data_frame("message_stats_table"),
                    ui.output_data_frame("temporal_stats_table"),
                    ui.output_data_frame("total_message_counts_table"),
                    full_screen=True
                )
            )
        elif q in [
            "Average Active Time per Module",
            "Average Active Time per Slide",
            "Unique Student Count per Slide",
            "Student Progression Graph"
        ]:
            return ui.layout_columns(
                ui.card(
                    ui.card_header("Main View"),
                    ui.output_plot("main_plot", height="650px"),
                    full_screen=True
                )
            )
        else:
            return ui.layout_columns(
                ui.card(
                    ui.card_header("Main Plot"),
                    ui.output_plot("main_plot", height="650px"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Data Preview"),
                    ui.output_data_frame("right_table"),
                    ui.output_data_frame("weekly_users_table"),
                    full_screen=True
                )
            )

    #main plots for all dropdown menus
    @output
    @render.plot
    def main_plot():
        q = input.question()
        df = current_df()
        dataset_choice = input.dataset_choice()


        if q == "Day Attempted Plot":
            if df.empty:
                fig, ax = plt.subplots()
                ax.set_title("No data available")
                ax.axis("off")
                return fig
            fig, day_df = plots.plot_day_attempted(df, input.dataset_choice())
            day_data_value.set(day_df)

            if dataset_choice == "New Kato" or dataset_choice == "New Kato July":
                weekly_users_df = compute_weekly_users(df)
                weekly_users_value.set(weekly_users_df)
            else:
                weekly_users_value.set(pd.DataFrame())
            
            return fig

        elif q == "Average Active Time per Module":
            gap_val = input.gap_threshold()
            return plots.plot_avg_time_per_module(data_map, gap_threshold=gap_val)

        elif q == "Average Active Time per Slide":
            dataset_choice = input.dataset_choice()
            if dataset_choice not in data_map:
                fig, ax = plt.subplots()
                ax.set_title("No dataset selected.")
                ax.axis("off")
                return fig

            gap_val = input.gap_threshold()
            return plots.plot_mean_time_per_slide(df, dataset_choice, gap_val)

        elif q == "Unique Student Count per Slide":
            dataset_choice = input.dataset_choice()
            if dataset_choice not in data_map:
                fig, ax = plt.subplots()
                ax.set_title("No dataset selected.")
                ax.axis("off")
                return fig
            return plots.plot_count_per_slide(df, dataset_choice)
        
        elif q == "Student Progression Graph":
            dataset_choice = input.dataset_choice()
            person = input.person()
            if dataset_choice not in data_map:
                fig, ax = plt.subplots()
                ax.set_title("No dataset selected.")
                ax.axis("off")
                return fig
            
            fig, slide_count = plots.plot_individual_completion(df, dataset_choice, person)
            slide_count_value.set(slide_count)
            return fig

        fig, ax = plt.subplots()
        ax.set_title("No question selected")
        ax.axis("off")
        return fig
    
    #Total message count table
    @output
    @render.data_frame
    def total_message_counts_table():
        if input.question() != "Conversation View":
            return pd.DataFrame()

        df = current_df()
        if df.empty:
            return pd.DataFrame()

        total_counts_df = compute_total_message_counts(df)
        return total_counts_df
    

    #conversation output UI elements
    @output
    @render.ui
    def conversation_output():
        if input.question() != "Conversation View":
            return None

        df = current_df()
        if df.empty:
            return ui.HTML("<p>No data available.</p>")

        df = df.dropna(subset=["ROLE", "CONTENT", "actor.name", "timestamp_gmt8"])

        chosen_person = input.person()
        if not chosen_person:
            return ui.HTML("<p>No person selected.</p>")

        user_df = df[df["actor.name"] == chosen_person].copy()
        if user_df.empty:
            return ui.HTML(f"<p>No conversation found for {chosen_person}.</p>")

        user_df["timestamp_gmt8"] = pd.to_datetime(user_df["timestamp_gmt8"])
        user_df = user_df.sort_values(by="timestamp_gmt8")

        lines = []
        for _, row in user_df.iterrows():
            timestamp_str = row["timestamp_gmt8"].strftime("%Y-%m-%d %H:%M:%S")
            role = str(row["ROLE"]).lower()
            content = row["CONTENT"]

            if role in ["user", "user-level2", "user-transcription"]:
                style = (
                    "display: inline-block;"
                    "max-width: 60%;"
                    "margin-left: auto;"
                    "margin-right: 20px;"
                    "text-align: right;"
                    "color: green;"
                    "background-color: #eaffea;"
                    "padding: 5px;"
                    "border-radius: 5px;"
                )

            elif role in ["preceptor", "preceptor-feedback", 
                        "preceptor-level2-feedback", 
                        "user-ddx-summary", 
                        "user-level0"]:
                style = (
                    "display: block;"
                    "text-align: center;"
                    "color: gray;"
                    "font-style: italic;"
                    "margin: 10px auto;" 
                    "max-width: 70%;"
                )

            elif role == "preceptor-level2":
                style = (
                    "display: inline-block;"
                    "max-width: 60%;"
                    "margin-right: auto;"
                    "margin-left: 20px;"
                    "text-align: left;"
                    "color: red;"
                    "background-color: #ffeeee;"
                    "padding: 5px;"
                    "border-radius: 5px;"
                )

            else:
                style = (
                    "display: inline-block;"
                    "max-width: 60%;"
                    "margin-right: auto;"
                    "margin-left: 20px;"
                    "text-align: left;"
                    "color: blue;"
                    "background-color: #eeeeff;"
                    "padding: 5px;"
                    "border-radius: 5px;"
                )

            line_html = f"""<div style="{style}"><strong>[{timestamp_str}] {role}:</strong> {content}</div>"""
            lines.append(line_html)

        html_text = "\n".join(lines)
        return ui.HTML(html_text)
    

    #renders conversation stats table in "conversation view"
    @output
    @render.data_frame
    def conv_stats_table():
        if input.question() != "Conversation View":
            return pd.DataFrame()

        df = current_df()
        if df.empty:
            return pd.DataFrame()
        
        total_users = df["actor.name"].nunique()
        df = df.dropna(subset=["ROLE","actor.name"])

        roles_of_interest = ["user-level0", "user", "patient", "preceptor", "preceptor-feedback", "user-level2", "preceptor-level2-feedback", "user-transcription"]
        results = []
        for role in roles_of_interest:
            num_users = df.loc[df["ROLE"] == role, "actor.name"].nunique()
            results.append({
                "role": role,
                "Distinct Users": num_users,
                "Percentage": str(float(f"{((num_users/total_users) * 100):.2f}")) + f"%"})

        table_df = pd.DataFrame(results)
        return table_df
    

    #shows each users chatbot time in minutes only if question == "conversation view"
    @output
    @render.data_frame
    def conv_avg_time_table():
        if input.question() != "Conversation View":
            return pd.DataFrame()

        df = current_df()
        if df.empty:
            return pd.DataFrame()

        table_df = compute_per_user_chatbot_time(df)
        return table_df
    

    #shows message stats table in conversation view by role
    @output
    @render.data_frame
    def message_stats_table():
        if input.question() != "Conversation View":
            return pd.DataFrame()

        df = current_df()
        if df.empty:
            return pd.DataFrame()

        stats_df = compute_message_stats(df)
        return stats_df


    #shows table on day attempted plot with percentages
    @output
    @render.data_frame
    def right_table():
        q = input.question()
        if q == "Day Attempted Plot":
            return day_data_value.get()
        else:
            return None


    #shows table of unique users per week
    @output
    @render.data_frame
    def weekly_users_table():
        q = input.question()
        dataset_choice = input.dataset_choice()
        
        if q != "Day Attempted Plot" or dataset_choice != "New Kato" or dataset_choice != "New Kato July":
            return None
        
        return weekly_users_value.get()

    #total student count for each dataset shown at the top
    @output
    @render.text
    def total_students():
        df = current_df()
        if df.empty:
            return "0"
        return str(df["actor.name"].nunique())

    #displays average slide time (precomputed in time_lookup_df24 / 23 to save on runtime.. you will need to adjust this)
    @output
    @render.text
    def avg_slide_time():
        dataset_choice = input.dataset_choice()

        if dataset_choice == "New Kato" or dataset_choice == "New Kato July":
            lookup_df = time_lookup_df24
        else:
            lookup_df = time_lookup_df23

        row = lookup_df[lookup_df["Module Name"] == dataset_choice]
        if row.empty:
            return "N/A"

        val = row["Average Slide Time (mins)"].values[0]
        return f"{val:.3f} minutes"
    
    #displays average completion time (precomputed in time_lookup_df24 / 23 to save on runtime.. you will need to adjust this)
    @output
    @render.text
    def avg_completion_time():
        dataset_choice = input.dataset_choice()
        
        if dataset_choice == "New Kato" or "New Kato July":
            lookup_df = time_lookup_df24
        else:
            lookup_df = time_lookup_df23

        row = lookup_df[lookup_df["Module Name"] == dataset_choice]
        if row.empty:
            return "N/A"

        val = row["Average Completion Time (mins)"].values[0]
        return f"{val:.3f} minutes"
    
    #which dataset to show which label
    @output
    @render.text
    def avg_slide_time_label():
        if input.dataset_choice() == "New Kato" or "New Kato July":
            return "Average Time on Chatbot"
        else:
            return "Average Slide Time"


    #calculates average chatbot time value for new dataset and provides std
    @output
    @render.text
    def avg_slide_time_value():
        dataset_choice = input.dataset_choice()

        if dataset_choice == "New Kato" or dataset_choice == "New Kato July":
            df = current_df()
            if df.empty:
                return "N/A"

            avg_time = compute_per_user_chatbot_time(df)
            avg_time_filtered = avg_time[avg_time["Chatbot Time (mins)"] > 0.5]
            
            if avg_time_filtered.empty:
                return "N/A"
            
            mean_time = avg_time_filtered["Chatbot Time (mins)"].mean()
            std_time = avg_time_filtered["Chatbot Time (mins)"].std()
            
            std_minutes = int(std_time)
            std_seconds = int((std_time - std_minutes) * 60)
            
            return f"{mean_time:.2f} minutes (std = ± {std_minutes}:{std_seconds:02d})"

        else:
            if dataset_choice == "Old Kato":
                lookup_df = time_lookup_df24
            else:
                lookup_df = time_lookup_df23

            row = lookup_df[lookup_df["Module Name"] == dataset_choice]
            if row.empty:
                return "N/A"
            val = row["Average Slide Time (mins)"].values[0]
            return f"{val:.1f} minutes"


www_dir = Path(__file__).parent.parent / "www"

app = App(app_ui, server, static_assets=www_dir)