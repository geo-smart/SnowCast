# Integrate all the datasets into one training dataset
import os.path
from datetime import datetime as dt

from snowcast_utils import *

pd.set_option('display.max_columns', None)

today = date.today()

# dd/mm/YY
start_date = "2022-01-01"
# end_date = today.strftime("%Y-%m-%d")
end_date = findLastStopDate(f"{github_dir}/data/sim_testing/gridmet/", "%Y-%m-%d %H:%M:%S")
print("d1 =", end_date)

print("integrating datasets into one dataset")
# pd.set_option('display.max_columns', None)

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
gridcells_file = f"{github_dir}/data/snowcast_provided/grid_cells.geojson"
model_dir = f"{github_dir}/model/"
training_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_train_features.csv"
testing_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_test_features.csv"
train_labels_file = f"{github_dir}/data/snowcast_provided/train_labels.csv"
ground_measure_metadata_file = f"{github_dir}/data/snowcast_provided/ground_measures_metadata.csv"
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"

# example_mod_file = f"{github_dir}/data/modis/mod10a1_ndsi_f191fe19-0e81-4bc9-9980-29738a05a49b.csv"


training_feature_pd = pd.read_csv(training_feature_file, header=0, index_col=0)
testing_feature_pd = pd.read_csv(testing_feature_file, header=0, index_col=0)
train_labels_pd = pd.read_csv(train_labels_file, header=0, index_col=0)
submission_format_pd = pd.read_csv(submission_format_file, header=0, index_col=0)
# print(training_feature_pd.head())

station_cell_mapper_pd = pd.read_csv(station_cell_mapper_file, header=0, index_col=0)


# print(station_cell_mapper_pd.head())

# example_mod_pd = pd.read_csv(example_mod_file, header=0, index_col=0)
# print(example_mod_pd.shape)
def getDateStr(x):
    return x.split(" ")[0]


def integrate_modis():
    """
    Integrate all MODIS data into mod_all.csv. Traverse all the csv files in the sat_testing/modis folder
    and aggregate them into one file with good headers.
    """
    all_mod_file = f"{github_dir}/data/ready_for_testing/modis_all.csv"
    ready_mod_file = f"{github_dir}/data/sat_testing/modis/mod10a1_ndsi_{start_date}_{end_date}.csv"
    mod_testing_folder = f"{github_dir}/data/sat_testing/modis/"
    if os.path.exists(all_mod_file):
        os.remove(all_mod_file)

    new_modis_pd = None

    for filename in os.listdir(mod_testing_folder):
        f = os.path.join(mod_testing_folder, filename)
        if os.path.isfile(f) and ".csv" in f:
            print(f)
            old_modis_pd = pd.read_csv(f, header=0)
            old_modis_pd = old_modis_pd.drop(columns=['date'])
            old_modis_pd.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
            # cell_id_list = old_modis_pd["cell_id"].unique()
            # cell_id_list = np.insert(cell_id_list, 0, "data")
            cell_id_list = submission_format_pd.index
            date_list = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)

            rows = date_list
            cols = cell_id_list

            if new_modis_pd is None:
                new_modis_pd = pd.DataFrame(([0.0 for col in cols] for row in rows), index=rows, columns=cols)

            for i, row in old_modis_pd.iterrows():
                cdate = row['date']
                ndsi = row['mod10a1_ndsi']
                cellid = row['cell_id']
                # print(f"{cdate} - {ndsi} - {cellid}")
                if ndsi != 0:
                    new_modis_pd.at[cdate, cellid] = ndsi

    # modis_np = numpy.zeros((len(date_list), len(cell_id_list)+1))
    # modis_np[0] = cell_id_list

    # s1_pd.loc[:, ~s1_pd.columns.str.match('Unnamed')]
    # print(new_modis_pd.head())
    new_modis_pd.to_csv(all_mod_file)


def integrate_sentinel1():
    """
    Integrate all Sentinel 1 data into sentinel1_all.csv
    Turn the rows into "daily", right now it has datetime stamps.
    """
    all_sentinel1_file = f"{github_dir}/data/ready_for_testing/sentinel1_all.csv"
    ready_sentinel1_file = f"{github_dir}/data/sat_testing/sentinel1/"
    if os.path.exists(all_sentinel1_file):
        os.remove(all_sentinel1_file)
    new_s1_pd = None
    for filename in os.listdir(ready_sentinel1_file):
        f = os.path.join(ready_sentinel1_file, filename)
        if os.path.isfile(f) and ".csv" in f:
            print(f)
            old_s1_pd = pd.read_csv(f, header=0)
            old_s1_pd = old_s1_pd.drop(columns=['date'])
            old_s1_pd.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
            # s1_pd.loc[:, ~s1_pd.columns.str.match('Unnamed')]

            # cell_id_list = old_s1_pd["cell_id"].unique()
            cell_id_list = submission_format_pd.index
            # date_list = old_s1_pd["date"].unique()
            date_list = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)
            rows = date_list
            cols = cell_id_list

            if new_s1_pd is None:
                new_s1_pd = pd.DataFrame(([0.0 for col in cols] for row in rows), index=rows, columns=cols)

            for i, row in old_s1_pd.iterrows():
                cdate = row['date']
                xdate = dt.strptime(cdate, "%Y-%m-%d %H:%M:%S")  # 3/7/2022  2:00:48 AM
                sdate = xdate.strftime("%Y-%m-%d")
                grd = row['s1_grd_vv']
                cellid = row['cell_id']
                if grd == 0:
                    continue
                new_s1_pd.at[sdate, cellid] = float(grd)

    new_s1_pd.to_csv(all_sentinel1_file)


def integrate_gridmet():
    """
    Integrate all gridMET data into gridmet_all.csv
    """

    dates = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)

    # print(mod_all_df.head())
    var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']

    for var in var_list:
        print("Processing ", var)
        all_single_var_file = f"{github_dir}/data/ready_for_testing/gridmet_{var}_all.csv"

        all_gridmet_var_folder = f"{github_dir}/data/sim_testing/gridmet/"
        new_var_pd = None

        for filename in os.listdir(all_gridmet_var_folder):
            f = os.path.join(all_gridmet_var_folder, filename)
            if os.path.isfile(f) and ".csv" in f:
                print(f)
                all_gridmet_var_pd = pd.read_csv(f, header=0)
                # cell_id_list = old_s1_pd["cell_id"].unique()
                cell_id_list = submission_format_pd.index
                # date_list = old_s1_pd["date"].unique()
                date_list = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)
                rows = date_list
                cols = cell_id_list
                if new_var_pd is None:
                    new_var_pd = pd.DataFrame(([0.0 for col in cols] for row in rows), index=rows, columns=cols)

                for i, row in all_gridmet_var_pd.iterrows():
                    cdate = row["Unnamed: 0"]
                    xdate = dt.strptime(cdate, "%Y-%m-%d %H:%M:%S")  # 3/7/2022  2:00:48 AM
                    sdate = xdate.strftime("%Y-%m-%d")
                    newval = row[var]
                    cellid = row['cell_id']
                    if newval != 0:
                        new_var_pd.at[sdate, cellid] = float(newval)

        new_var_pd.to_csv(all_single_var_file)


def prepare_testing_csv():
    """
    MOD model:
      input columns: [m, doy, ndsi]
      output column: [swe]
    Sentinel1 model:
      input columns: [m, doy, grd]
      output column: [swe]
    gridMET model:
      input columns: [m, doy, tmmn, tmmx, pr, vpd, eto, rmax, rmin, vs]
      output column: [swe]
    """
    all_ready_file = f"{github_dir}/data/ready_for_testing/all_ready_3.csv"
    if os.path.exists(all_ready_file):
        os.remove(all_ready_file)

    all_mod_file = f"{github_dir}/data/ready_for_testing/modis_all.csv"
    modis_all_pd = pd.read_csv(all_mod_file, header=0, index_col=0)
    modis_all_np = modis_all_pd.to_numpy()

    all_sentinel1_file = f"{github_dir}/data/ready_for_testing/sentinel1_all.csv"
    sentinel1_all_pd = pd.read_csv(all_sentinel1_file, header=0, index_col=0)
    sentinel1_all_np = sentinel1_all_pd.to_numpy()

    all_gridmet_eto_file = f"{github_dir}/data/ready_for_testing/gridmet_eto_all.csv"
    gridmet_eto_all_pd = pd.read_csv(all_gridmet_eto_file, header=0, index_col=0)
    gridmet_eto_all_np = gridmet_eto_all_pd.to_numpy()

    all_gridmet_pr_file = f"{github_dir}/data/ready_for_testing/gridmet_pr_all.csv"
    gridmet_pr_all_pd = pd.read_csv(all_gridmet_pr_file, header=0, index_col=0)
    gridmet_pr_all_np = gridmet_pr_all_pd.to_numpy()

    all_gridmet_rmax_file = f"{github_dir}/data/ready_for_testing/gridmet_rmax_all.csv"
    gridmet_rmax_all_pd = pd.read_csv(all_gridmet_rmax_file, header=0, index_col=0)
    gridmet_rmax_all_np = gridmet_rmax_all_pd.to_numpy()

    all_gridmet_rmin_file = f"{github_dir}/data/ready_for_testing/gridmet_rmin_all.csv"
    gridmet_rmin_all_pd = pd.read_csv(all_gridmet_rmin_file, header=0, index_col=0)
    gridmet_rmin_all_np = gridmet_rmin_all_pd.to_numpy()

    all_gridmet_tmmn_file = f"{github_dir}/data/ready_for_testing/gridmet_tmmn_all.csv"
    gridmet_tmmn_all_pd = pd.read_csv(all_gridmet_tmmn_file, header=0, index_col=0)
    gridmet_tmmn_all_np = gridmet_tmmn_all_pd.to_numpy()

    all_gridmet_tmmx_file = f"{github_dir}/data/ready_for_testing/gridmet_tmmx_all.csv"
    gridmet_tmmx_all_pd = pd.read_csv(all_gridmet_tmmx_file, header=0, index_col=0)
    gridmet_tmmx_all_np = gridmet_tmmx_all_pd.to_numpy()

    all_gridmet_vpd_file = f"{github_dir}/data/ready_for_testing/gridmet_vpd_all.csv"
    gridmet_vpd_all_pd = pd.read_csv(all_gridmet_vpd_file, header=0, index_col=0)
    gridmet_vpd_all_np = gridmet_vpd_all_pd.to_numpy()

    all_gridmet_vs_file = f"{github_dir}/data/ready_for_testing/gridmet_vs_all.csv"
    gridmet_vs_all_pd = pd.read_csv(all_gridmet_vs_file, header=0, index_col=0)
    gridmet_vs_all_np = gridmet_vs_all_pd.to_numpy()

    grid_terrain_file = f"{github_dir}/data/terrain/gridcells_eval_terrainData.csv"
    grid_terrain_pd = pd.read_csv(grid_terrain_file, header=0, index_col=0)
    grid_terrain_np = grid_terrain_pd.to_numpy()

    sentinel1_all_pd = sentinel1_all_pd[:modis_all_pd.shape[0]]

    print("modis_all_size: ", modis_all_pd.shape)
    print("sentinel1_all_size: ", sentinel1_all_pd.shape)
    print("gridmet rmax size: ", gridmet_rmax_all_pd.shape)
    print("gridmet eto size: ", gridmet_eto_all_pd.shape)
    print("gridmet vpd size: ", gridmet_vpd_all_pd.shape)
    print("gridmet pr size: ", gridmet_pr_all_pd.shape)
    print("gridmet rmin size: ", gridmet_rmin_all_pd.shape)
    print("gridmet tmmn size: ", gridmet_tmmn_all_pd.shape)
    print("gridmet tmmx size: ", gridmet_tmmx_all_pd.shape)
    print("gridmet vs size: ", gridmet_vs_all_pd.shape)
    print("grid terrain size: ", grid_terrain_pd.shape)
    print("cell_size: ", len(submission_format_pd.index))
    print("station size: ", station_cell_mapper_pd.shape)
    print("training_feature_pd size: ", training_feature_pd.shape)
    print("testing_feature_pd size: ", testing_feature_pd.shape)
    print("grid_terrain_np shape: ", grid_terrain_np.shape)

    min_len = min(modis_all_pd.shape[0], sentinel1_all_pd.shape[0], gridmet_rmax_all_pd.shape[0],
                  gridmet_eto_all_pd.shape[0], gridmet_vpd_all_pd.shape[0], gridmet_pr_all_pd.shape[0],
                  gridmet_rmin_all_pd.shape[0], gridmet_tmmn_all_pd.shape[0], gridmet_tmmx_all_pd.shape[0],
                  gridmet_vs_all_pd.shape[0], grid_terrain_pd.shape[0])

    cell_id_list = modis_all_pd.columns.values

    # create a multiple numpy array, the dimension is (cell_id, date, variable)
    # all_testing_np = np.empty((len(modis_all_pd.index.values), len(modis_all_pd.columns.values),  23))
    all_testing_np = np.empty((min_len, len(modis_all_pd.columns.values), 23))
    print("final all numpy shape: ", all_testing_np.shape)

    modis_all_np = np.expand_dims(modis_all_np[:min_len, :], axis=2)
    sentinel1_all_np = np.expand_dims(sentinel1_all_np[:min_len, :], axis=2)
    gridmet_eto_all_np = np.expand_dims(gridmet_eto_all_np[:min_len, :], axis=2)
    gridmet_pr_all_np = np.expand_dims(gridmet_pr_all_np[:min_len, :], axis=2)
    gridmet_rmax_all_np = np.expand_dims(gridmet_rmax_all_np[:min_len, :], axis=2)
    gridmet_rmin_all_np = np.expand_dims(gridmet_rmin_all_np[:min_len, :], axis=2)
    gridmet_tmmn_all_np = np.expand_dims(gridmet_tmmn_all_np[:min_len, :], axis=2)
    gridmet_tmmx_all_np = np.expand_dims(gridmet_tmmx_all_np[:min_len, :], axis=2)
    gridmet_vpd_all_np = np.expand_dims(gridmet_vpd_all_np[:min_len, :], axis=2)
    gridmet_vs_all_np = np.expand_dims(gridmet_vs_all_np[:min_len, :], axis=2)

    cell_id_np = np.expand_dims(cell_id_list, axis=0)
    cell_id_np = np.repeat(cell_id_np, min_len, axis=0)
    cell_id_np = np.expand_dims(cell_id_np, axis=2)
    print("cell_id_np shape: ", cell_id_np.shape)

    grid_terrain_np = np.expand_dims(grid_terrain_np, axis=0)
    grid_terrain_np = np.repeat(grid_terrain_np, min_len, axis=0)

    date_np = np.empty((min_len, len(modis_all_pd.columns.values), 3))
    for i in range(min_len):
        # print(i, " - ", modis_all_pd.index.values[i])
        date_time_obj = dt.strptime(modis_all_pd.index.values[i], '%Y-%m-%d')
        date_np[i, :, 0] = date_time_obj.year
        date_np[i, :, 1] = date_time_obj.month
        date_np[i, :, 2] = date_time_obj.timetuple().tm_yday

    new_np = np.concatenate((cell_id_np, date_np, modis_all_np, sentinel1_all_np, gridmet_eto_all_np, gridmet_pr_all_np,
                             gridmet_rmax_all_np, gridmet_rmin_all_np, gridmet_tmmn_all_np, gridmet_tmmx_all_np,
                             gridmet_vpd_all_np, gridmet_vs_all_np, grid_terrain_np), axis=2)
    print("new numpy shape: ", new_np.shape)

    new_np = new_np.reshape(-1, new_np.shape[-1])
    print("reshaped: ", new_np.shape)

    # all_training_pd = pd.DataFrame(columns=["cell_id", "year", "m", "doy", "ndsi", "grd", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "lat", "lon", "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe"])
    all_testing_pd = pd.DataFrame(new_np,
                                  columns=["cell_id", "year", "m", "doy", "ndsi", "grd", "eto", "pr", "rmax", "rmin",
                                           "tmmn", "tmmx", "vpd", "vs", "lat", "lon", "elevation", "aspect",
                                           "curvature", "slope", "eastness", "northness"])

    # print("MODIS all np shape: ", modis_all_np.shape)
    # print("Terrain numpy shape: ", grid_terrain_np.shape)

    # print("Head", all_testing_pd.head())
    all_testing_pd.to_csv(all_ready_file)


# exit() # done already

integrate_modis()
integrate_sentinel1()
integrate_gridmet()
prepare_testing_csv()
