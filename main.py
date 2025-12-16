import pandas as pd
from data_access.data_access_services import Data_access_services
from services import fichier_enquetes_to_agreg_sites_services as agreg_services
from services import prepa_data_services
from services import modeles_services
from services import metrics_services
from services import cross_validation_services
from services import simul_agreg_test

from matplotlib import pyplot as plt
import numpy as np
import math
import sys


obj_data_access_services=Data_access_services()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



nom_table_enquetes_agregees="enquetes_maisons"
schema= 'points_enquete'

LOAD_anf_AGGREG_DONNEES_ENQUETE_VF=False
LOAD_DONNEES_AGREGEES_VF=False
LOAD_DONNEES_BATIMENTS_VF=False
#Les données maisons->bâtiments->cosia sont calculées
LOAD_RAW_FEATURES_VF=True



if LOAD_anf_AGGREG_DONNEES_ENQUETE_VF:
    TEST_MULTI_VF=False
    AGREGATION_VF=True
    distance_max_pour_agreg_en_m=200.0
    EXPORT_TO_BD_VF=True
    agreg_services.enquetes_to_agreg_sites(TEST_MULTI_VF=TEST_MULTI_VF,AGREGATION_VF=AGREGATION_VF,distance_max_pour_agreg_en_m=distance_max_pour_agreg_en_m,EXPORT_TO_BD_VF=EXPORT_TO_BD_VF,nom_table_enquetes_agregees=nom_table_enquetes_agregees,schema=schema)

if LOAD_DONNEES_AGREGEES_VF:
    print(f'LOAD TABLE {nom_table_enquetes_agregees} from BD')
    gpd_maisons=obj_data_access_services.get_data_table_to_gdf(tablename=nom_table_enquetes_agregees)
    #
    print(f'\nTABLE {nom_table_enquetes_agregees} LOADED')
    print(f'->{len(gpd_maisons)} ligne(s) \n')
    print(gpd_maisons.head(1))

if LOAD_DONNEES_BATIMENTS_VF:
   
    table='donnees_batiments'
    schema='traitements'
    col_geometry='geom'
    
    gpd_batiments=obj_data_access_services.get_data_table_to_gdf(tablename=table,col_geometry=col_geometry,schema=schema)
    print(gpd_batiments.head(1))
    
if LOAD_RAW_FEATURES_VF:
    table='raw_features'
    schema='points_enquete'
    col_geometry='geometry'
    nb_maxi=-1

    gpd_raw_features=obj_data_access_services.get_data_table_to_gdf(tablename=table,col_geometry=col_geometry,schema=schema,nb_maxi=nb_maxi)
    print(f'RAW FEATURES:')
    print(f'->{len(gpd_raw_features)} ligne(s)')
    print(f'->col: \n{gpd_raw_features.columns}')
   
    
    gpd_filtered_features=gpd_raw_features.copy()

    columns_to_drop = [
    'id_maison_b50_m',
    'id_batiment_b50_m',
    'surf_batiment_source_m2_b50_m']

    gpd_filtered_features = gpd_filtered_features.drop(
    columns=columns_to_drop,
    errors='ignore'  # évite une erreur si une colonne est absente
    )
    gpd_filtered_features.rename(columns={'id_batiment_b10_m':'id_batiment','surf_batiment_source_m2_b10_m':'surf_batiment_source_m2'},inplace=True)
    #
    gpd_filtered_features=gpd_filtered_features[gpd_filtered_features['visite_VF']==True]
    gpd_filtered_features=gpd_filtered_features[(gpd_filtered_features['contenant enterré']>=0) & (gpd_filtered_features['grand contenant']>=0) & (gpd_filtered_features['petit contenant']>=0)]

    print(f'\nFILTERED FEATURES:')
    print(f'->{len(gpd_filtered_features)} ligne(s)')
    print(f'->col: \n{gpd_filtered_features.columns}')
    #print(gpd_filtered_features.head(1))

    print(gpd_filtered_features[['contenant enterré','grand contenant', 'petit contenant']].describe())



def plot_boxplots_features(
    df,
    features,
    figsize=(8, 8),
    title=None,
    showfliers=True
):
    """
    Display vertical boxplots for selected integer features.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Input dataframe.
    features : list of str
        Column names to plot.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Global title.
    showfliers : bool, optional
        Whether to display outliers.
    """

    fig, axes = plt.subplots(
        nrows=len(features),
        ncols=1,
        figsize=figsize,
        sharex=False
    )

    # Ensure axes is iterable even for a single feature
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        data = df[feature].dropna()

        ax.boxplot(
            data,
            vert=False,
            showfliers=showfliers
        )

        ax.set_title(feature)
        ax.set_xlabel("Value (integer count)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.92)

    plt.tight_layout()
    plt.show()



    # Liste des colonnes à afficher
features = [
    'contenant enterré',
    'grand contenant',
    'petit contenant'
]

visu_boxplots_vf=False
visu_freq_p95_vf=False

if visu_boxplots_vf:
    plot_boxplots_features(
        df=gpd_filtered_features,
        features=features,
        title="Distribution des contenants – Boxplots",
        showfliers=True
    )

if visu_freq_p95_vf:
    #Création de la figure avec 3 sous-graphiques verticaux
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 10))

    for ax, feature in zip(axes, features):
        data = gpd_filtered_features[feature]

        # Quantile 0.95
        q95 = data.quantile(0.95)
        max_bin = math.ceil(q95)

        # Bins entiers
        bins = np.arange(0, max_bin + 2)  # +2 pour inclure la dernière classe

        ax.hist(
            data[data <= max_bin],
            bins=bins,
            edgecolor='black',
            align='left'
            )

        ax.set_title(f"{feature} (jusqu’au 95e percentile)")
        ax.set_xlabel("Nombre (valeurs entières)")
        ax.set_ylabel("Fréquence")

        ax.set_xticks(bins[:-1])

    plt.tight_layout()
    plt.show()



features_col=[
       'id_maison',
       'surf_batiment_source_m2','hauteur',
       'surf_buffer_m2_b10_m', 'surf_batiment_b10_m', 'surf_broussaille_b10_m',
       'surf_conifere_b10_m', 'surf_feuillu_b10_m', 'surf_pelouse_b10_m',
       'surf_piscine_b10_m', 'surf_serre_b10_m', 'surf_sol_nu_b10_m',
       'surf_surface_eau_b10_m', 'surf_zone_impermeable_b10_m',
       'surf_zone_permeable_b10_m', 'surf_buffer_m2_b50_m',
       'surf_batiment_b50_m', 'surf_broussaille_b50_m', 'surf_conifere_b50_m',
       'surf_feuillu_b50_m', 'surf_pelouse_b50_m', 'surf_piscine_b50_m',
       'surf_serre_b50_m', 'surf_sol_nu_b50_m', 'surf_surface_eau_b50_m',
       'surf_zone_impermeable_b50_m', 'surf_zone_permeable_b50_m']

targets_col=['id_maison', 'contenant enterré',
       'grand contenant', 'petit contenant', 'terrasses sur plots',
       'nb_conteneurs_total' ]


row_nb=len(gpd_filtered_features)
print('Nb de valeurs 0:')
for f in features:
    c0=len(gpd_filtered_features[gpd_filtered_features[f]==0])
    print(f'{f}: {c0} / {row_nb}')

TEST_NULLS_VF=False
if TEST_NULLS_VF:
    null_counts = gpd_filtered_features.isna().sum()
    null_percent = (
        gpd_filtered_features.isna().mean() * 100
    ).round(2)
    print(f'->valeurs nulles: {null_counts}')
    print(f'->valeurs nulles pc: {null_percent}')
    print(null_counts[null_counts > 0].sort_values(ascending=False))
    print(null_percent[null_percent > 0].sort_values(ascending=False))

DEFAULT_HEIGHT = 5.0

height_effective = (
    gpd_filtered_features["hauteur"]
    .astype(float)
    .fillna(DEFAULT_HEIGHT)
)
gpd_filtered_features["hauteur_corrigee_m"]=height_effective

features_col.remove('hauteur')
features_col=features_col+["hauteur_corrigee_m"]

service = prepa_data_services.BuildingsCountDataPreparationService(id_col="id_maison")

artifacts = service.prepare_dataset(
    df=gpd_filtered_features,
    features_col=features_col,
    targets_col=targets_col,
    add_volume=True,
    make_ratios=True,
    add_log1p=True,
    target_cols_to_keep=["contenant enterré", "grand contenant", "petit contenant"],
)

X, Y, ids = artifacts.X, artifacts.Y, artifacts.ids

print(f'->colonnes x:\n {X.columns}')
#sys.exit()
# print(f'->colonnes y:\n {Y.columns}')


#On peut enlever les surfaces brutes:
X = artifacts.X.copy()

drop_cols = [c for c in X.columns
             if (c.startswith("surf_") and ("_b10_m" in c or "_b50_m" in c)
                 and not c.startswith("surf_buffer_m2_"))]

X = X.drop(columns=drop_cols)



#TEST BASE LINE:
target = "contenant enterré"
y_true = Y[target]

baseline_A =modeles_services.BaselineMeanPredictor().fit(y_true)
y_pred_A = baseline_A.predict(len(y_true))

metrics_A = metrics_services.CountRegressionMetrics.compute_all(y_true, y_pred_A)

print('Metrics sur baseline mean:')
print(metrics_A)


#
baseline_B_features = [
    "log1p_surf_batiment_source_m2",
    "hauteur_corrigee_m",
    "log1p_volume_batiment",
    'log1p_surf_buffer_m2_b10_m', 'log1p_surf_buffer_m2_b50_m'
]





baseline_B = modeles_services.BaselineNegativeBinomial(feature_cols=baseline_B_features)
baseline_B.fit(X, y_true)

y_pred_B = baseline_B.predict(X)

metrics_B = metrics_services.CountRegressionMetrics.compute_all(y_true, y_pred_B)

print('Metrics sur baseline regression binomiale negative:')
print(metrics_B)


cv_results = cross_validation_services.cross_validate_baselines(
    X=X,
    y=Y[target],
    baseline_B_features=baseline_B_features,
)
print(cv_results)

summary = (
    cv_results
    .groupby("baseline")
    .agg(["mean", "std"])
)

print(summary)


agg_results = simul_agreg_test.cv_aggregated_protocol(
    X=X,
    y=Y[target],
    baseline_B_features=baseline_B_features,
    group_sizes=(30, 60, 120),
    n_draws=1000,
)
print(agg_results)

summary = (
    agg_results
    .groupby(["model", "group_size"])
    .agg(["mean", "std"])
)

print(summary)