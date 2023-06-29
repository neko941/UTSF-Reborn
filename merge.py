import polars as pl
# import os
# from rich.progress import Progress

# # Define the paths to the directories
# path1 = r".\data\raw\traffic\a_year-min_max_std"
# path2 = r".\data\raw\traffic\a_year-has_rain"

# # Initialize empty DataFrames to store the data
# data_frame1 = pl.DataFrame()
# data_frame2 = pl.DataFrame()

# # Iterate over the CSV files in path1
# with Progress() as progress:
#     task1 = progress.add_task("[cyan]Reading path1...", total=len(os.listdir(path1)))
#     for filename in os.listdir(path1):
#         if filename.endswith(".csv"):
#             file_path = os.path.join(path1, filename)
#             df = pl.read_csv(source=file_path, separator='|', has_header=True, try_parse_dates=True, low_memory=True, infer_schema_length=10000)
#             data_frame1 = pl.concat([data_frame1, df])
#             progress.update(task1, advance=1)
# data_frame1.write_csv('min_max_std.csv', separator='|')
# # data_frame1 = pl.read_csv(source='min_max_std.csv', separator='|', has_header=True, try_parse_dates=True, low_memory=True, infer_schema_length=10000)

# # # Iterate over the CSV files in path2
# with Progress() as progress:
#     task2 = progress.add_task("[cyan]Reading path2...", total=len(os.listdir(path2)))
#     for filename in os.listdir(path2):
#         if filename.endswith(".csv"):
#             file_path = os.path.join(path2, filename)
#             df = pl.read_csv(source=file_path, separator='|', has_header=True, try_parse_dates=True, low_memory=True, infer_schema_length=10000)
#             df = df.with_columns(pl.col('rain').cast(pl.Float64))
#             data_frame2 = pl.concat([data_frame2, df])
#             progress.update(task2, advance=1)
# data_frame2.write_csv('has_rain.csv', separator='|')
# # data_frame2 = pl.read_csv(source='has_rain.csv', separator='|', has_header=True, try_parse_dates=True, low_memory=True, infer_schema_length=10000)

# # Merge the data based on common columns
# merged_data = data_frame1.join(data_frame2, on=["date", "time_id", "kml_segment_id"])

# # Print the merged data
# print(merged_data)

# # Save the merged data to CSV
# output_file = "merged_data.csv"
# merged_data.write_csv(output_file, separator='|')

# # Print a success message
# print(f"Merged data saved to {output_file}")


df = pl.read_csv(source="merged_data.csv", separator='|', has_header=True, try_parse_dates=True, low_memory=True, infer_schema_length=10000)
import polars as pl
print(df.filter(pl.col("speed") != pl.col("avg_speed")))