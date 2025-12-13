import pandas as pd
from data_access.data_access_services import Data_access_services
from services import fichier_enquetes_to_agreg_sites_services as agreg_services
from matplotlib import pyplot as plt
import numpy as np
import math


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
  
    #
    col_utiles=['id_maison', 'date_visite', 'code_agent', 'x_visite', 'y_visite',
       'code_insee', 'type_enquete', 'visite_VF', 'contenant enterré',
       'grand contenant', 'petit contenant', 'terrasses sur plots',
       'nb_conteneurs_total', 'commentaires', 'geometry', 'cluster_200m',
       'id_maison_b10_m', 'id_batiment_b10_m', 'surf_batiment_source_m2_b10_m',
       'surf_buffer_m2_b10_m', 'surf_batiment_b10_m', 'surf_broussaille_b10_m',
       'surf_conifere_b10_m', 'surf_feuillu_b10_m', 'surf_pelouse_b10_m',
       'surf_piscine_b10_m', 'surf_serre_b10_m', 'surf_sol_nu_b10_m',
       'surf_surface_eau_b10_m', 'surf_zone_impermeable_b10_m',
       'surf_zone_permeable_b10_m', 'surf_buffer_m2_b50_m',
       'surf_batiment_b50_m', 'surf_broussaille_b50_m', 'surf_conifere_b50_m',
       'surf_feuillu_b50_m', 'surf_pelouse_b50_m', 'surf_piscine_b50_m',
       'surf_serre_b50_m', 'surf_sol_nu_b50_m', 'surf_surface_eau_b50_m',
       'surf_zone_impermeable_b50_m', 'surf_zone_permeable_b50_m']
    
    gpd_filtered_features=gpd_raw_features[col_utiles].copy()
    gpd_filtered_features.rename(columns={'id_batiment_b10_m':'id_batiment'},inplace=True)
    #
    gpd_filtered_features=gpd_filtered_features[gpd_filtered_features['visite_VF']==True]
    gpd_filtered_features=gpd_filtered_features[(gpd_filtered_features['contenant enterré']>=0) & (gpd_filtered_features['grand contenant']>=0) & (gpd_filtered_features['petit contenant']>=0)]

    print(f'\nFILTERED FEATURES:')
    print(f'->{len(gpd_filtered_features)} ligne(s)')
    print(f'->col: \n{gpd_filtered_features.columns}')
    #print(gpd_filtered_features.head(1))

    print(gpd_filtered_features[['contenant enterré','grand contenant', 'petit contenant']].describe())

    # Liste des colonnes à afficher
features = [
    'contenant enterré',
    'grand contenant',
    'petit contenant'
]



#Création de la figure avec 3 sous-graphiques verticaux
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

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

 

    

