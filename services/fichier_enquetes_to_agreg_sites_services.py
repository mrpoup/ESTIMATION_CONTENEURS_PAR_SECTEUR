import pandas as pd
import numpy as np
from pyproj import Transformer
import geopandas as gpd
from sklearn.cluster import DBSCAN
from data_access.data_access_services import Data_access_services

def enquetes_to_agreg_sites(TEST_MULTI_VF=False,AGREGATION_VF=True,distance_max_pour_agreg_en_m=200.0,EXPORT_TO_BD_VF=True,nom_table_enquetes_agregees="enquetes_maisons",schema= 'points_enquete'):

    obj_data_access_services=Data_access_services()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)


    REMONTE_et_AGREGE_DONNEES_ENQUETE_VF=True
    REMONTE_DONNEES_AGREGEES_VF=False

    print('REMONTEE des DONNEES BRUTES:')
    nb_lignes_maxi_moins1_si_toutes=-1
    raw_data=obj_data_access_services.get_gites_raw_file(nb_maxi=nb_lignes_maxi_moins1_si_toutes,schema=schema)
    #print(f'Fichier brut colonnes:\n{raw_data.columns}')

    col_utiles=['_uid_', 'id_maison', 'date_visite','code_agent',
                'x_visite', 'y_visite','x_saisie', 'y_saisie','code_insee',
                'origine','type',
                'nuisance','nb_moustiqr','lb_ctn','qte_ctn','commentaires'
                ]
    ren_col={'_uid_':'id_ligne','origine':'type_enquete','type':'avancement','lb_ctn':'type_conteneur','qte_ctn':'nb_conteneurs'}

    df_raw_data_filtered=raw_data[col_utiles]
    df_raw_data_filtered.rename(columns=ren_col,inplace=True)
    print(df_raw_data_filtered.head(5))
    print(f'COLONNES:\n {df_raw_data_filtered.columns}')


    print(f'\nTYPES ENQUETES:\n {df_raw_data_filtered['type_enquete'].value_counts()}')
    # type_enquete
    # Mail/courrier                6740
    # Porte à porte                4985
    # Enquete entomo                652
    # Signalement moustique         575
    # Porte Ã  porte                451
    # Appel à l'EID pr plainte      387
    # Appel Ã  l'EID pr plainte      88

    def normalize(text):
        if not isinstance(text, str):
            return text
        return (
            text.encode("latin1")  # interprète les caractères 'Ã' comme Latin-1
                .decode("utf8", errors="ignore")  # et les convertit en UTF-8 correct
                .strip()  # retire espaces parasites
        )

    df_raw_data_filtered["type_enquete"] = df_raw_data_filtered["type_enquete"].apply(normalize)

    print(f'\nTYPES ENQUETES NEW:\n {df_raw_data_filtered['type_enquete'].value_counts()}')
    # TYPES ENQUETES NEW:
    # type_enquete
    # Mail/courrier               6740
    # Porte  porte                4985
    # Enquete entomo               652
    # Signalement moustique        575
    # Porte à porte                451
    # Appel  l'EID pr plainte      387
    # Appel à l'EID pr plainte      88

    ren_type_enquete={}
    ren_type_enquete['Mail/courrier']='plainte'
    ren_type_enquete['Signalement moustique']='plainte'
    ren_type_enquete["Appel à l'EID pr plainte"]='plainte'
    ren_type_enquete["Appel  l'EID pr plainte"]='plainte'
    ren_type_enquete['Porte  porte']='porte_a_porte'
    ren_type_enquete['Porte Ã  porte']='porte_a_porte'
    ren_type_enquete['Enquete entomo']='enquete_entomo'

    df_raw_data_filtered['type_enquete']=df_raw_data_filtered['type_enquete'].replace(ren_type_enquete)
    print(f'\nTYPES ENQUETES NEW:\n {df_raw_data_filtered['type_enquete'].value_counts()}')
    # TYPES ENQUETES NEW:
    #  type_enquete
    # plainte           7790
    # porte_a_porte     4985
    # enquete_entomo     652
    # Porte à porte      451

    print(f'\nTYPES CONTENEURS:\n {df_raw_data_filtered['type_conteneur'].value_counts()}')
    # type_conteneur
    # contenant enterré      5261
    # grand contenant        4518
    # petit contenant        3904
    # terrasses sur plots     195

    print(f'\nTYPES AVANCEMENTS ENQUETES:\n {df_raw_data_filtered['avancement'].value_counts()}')

    df_raw_data_filtered['visite_VF']=df_raw_data_filtered['avancement']=='visite'

    print(f'\nENQUETES OK KO:\n {df_raw_data_filtered['visite_VF'].value_counts()}')


    print('COORD VISITE:')
    coord_visite=df_raw_data_filtered[['x_visite', 'y_visite']]
    print(coord_visite.head(5))
    #->lambert 93
    print(coord_visite.describe())



    coord_saisie=df_raw_data_filtered[(df_raw_data_filtered['x_saisie'] !=0) & (df_raw_data_filtered['y_saisie'] !=0)].copy()
    print('COORD SAISIE:')
    print(coord_saisie[['x_saisie','y_saisie']])
    #->lat long!

    #->Je convertis en lambert 93
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_l93, y_l93 = transformer.transform(
        coord_saisie["x_saisie"].values,
        coord_saisie["y_saisie"].values
    )
    coord_saisie['x_saisie_l93']=x_l93.round(1)
    coord_saisie['y_saisie_l93']=y_l93.round(1)

    #->Je "mixe" les 2 champs
    df_raw_data_filtered['x_saisie_l93']=df_raw_data_filtered['x_visite']
    df_raw_data_filtered['y_saisie_l93']=df_raw_data_filtered['y_visite']
    df_raw_data_filtered.loc[coord_saisie.index, "x_saisie_l93"] = coord_saisie["x_saisie_l93"]
    df_raw_data_filtered.loc[coord_saisie.index, "y_saisie_l93"] = coord_saisie["y_saisie_l93"]

    df_raw_data_filtered['distance_visite_saisie_m'] = np.sqrt(
        (df_raw_data_filtered['x_saisie_l93'] - df_raw_data_filtered['x_visite'])**2 +
        (df_raw_data_filtered['y_saisie_l93'] - df_raw_data_filtered['y_visite'])**2
    ).round(0)

    ecart_max=20000
    t=df_raw_data_filtered[df_raw_data_filtered['distance_visite_saisie_m']>ecart_max][['x_visite','y_visite','x_saisie_l93','y_saisie_l93','distance_visite_saisie_m']].sort_values('distance_visite_saisie_m',ascending=False )
    print(f'->Ecart max: {df_raw_data_filtered['distance_visite_saisie_m'].max()}')
    #->Ecart max: 25112.0
    print(f'\nECARTS DISTANCE VISITE vs SAISIE >{ecart_max}->{len(t)}')

    #ECARTS DISTANCE VISITE vs SAISIE >0->222
    #ECARTS DISTANCE VISITE vs SAISIE >1000->88
    #ECARTS DISTANCE VISITE vs SAISIE >10000->47
    #ECARTS DISTANCE VISITE vs SAISIE >20000->19


    #VERIF SITES!
    df_raw_data_filtered["code_coord_visite"] = (
        df_raw_data_filtered["x_visite"].round(0).astype(int).astype(str)
        + "_"
        + df_raw_data_filtered["y_visite"].round(0).astype(int).astype(str)
    )

    df_raw_data_filtered["code_coord_saisie_l93"] = (
        df_raw_data_filtered["x_saisie_l93"].round(0).astype(int).astype(str)
        + "_"
        + df_raw_data_filtered["x_saisie_l93"].round(0).astype(int).astype(str)
    )



    #____________________________________________________


    if TEST_MULTI_VF:

        coord_par_id_maison = (
            df_raw_data_filtered
            .groupby("id_maison")["code_coord_visite"]
            .unique()
            .apply(lambda x: sorted(x))
            .to_dict()
        )

        v_multi={c:v for c,v in coord_par_id_maison.items() if len(v)>1}
        print(f'maison multi_coord: {len(v_multi)}')
        print(v_multi)
        #->pas de maisons avec des coordonnées multiples.



        #VERIF SITES SAISIE L93

        coord_saisie_par_id_maison = (
            df_raw_data_filtered
            .groupby("id_maison")["code_coord_saisie_l93"]
            .unique()
            .apply(lambda x: sorted(x))
            .to_dict()
        )

        v_multi={c:v for c,v in coord_saisie_par_id_maison.items() if len(v)>1}
        print(f'maison multi coord saisie: {len(v_multi)}')
        print(v_multi)
        #->pas de maisons avec des coordonnées multiples.
        dict_type_enquetes_par_id_maison = (
            df_raw_data_filtered
            .groupby("id_maison")["type_enquete"]
            .unique()
            .apply(lambda x: sorted(x))
            .to_dict()
        )
        v_multi={c:v for c,v in dict_type_enquetes_par_id_maison.items() if len(v)>1}
        print(f'enquetes multi: {len(v_multi)}')


        dict_date_visite_par_id_maison = (
            df_raw_data_filtered
            .groupby("id_maison")["date_visite"]
            .unique()
            .apply(lambda x: sorted(x))
            .to_dict()
        )
        v_multi={c:v for c,v in dict_date_visite_par_id_maison.items() if len(v)>1}
        print(f'dates visite multi: {len(v_multi)}')
        print(v_multi)

        dict_agent_par_id_maison = (
            df_raw_data_filtered
            .groupby("id_maison")["code_agent"]
            .unique()
            .apply(lambda x: sorted(x))
            .to_dict()
        )
        v_multi={c:v for c,v in dict_date_visite_par_id_maison.items() if len(v)>1}
        print(f'agent multi: {len(v_multi)}')
        print(v_multi)


        dict_avancement_par_id_maison = (
            df_raw_data_filtered
            .groupby("id_maison")["avancement"]
            .unique()
            .apply(lambda x: sorted(x))
            .to_dict()
        )
        v_multi={c:v for c,v in dict_avancement_par_id_maison.items() if len(v)>1}
        print(f'avancement multi: {len(v_multi)}')
        print(v_multi)


    #AGREGATION

    if AGREGATION_VF:
        df_conteneurs_par_maison = (
            df_raw_data_filtered
            .pivot_table(
                index="id_maison",              # 1 ligne par maison
                columns="type_conteneur",       # 1 colonne par type de conteneur
                values="nb_conteneurs",         # valeurs numériques
                aggfunc="sum",                  # on somme si plusieurs lignes de même type
                fill_value=0                    # met 0 quand il n'y a pas de valeur
            )
            .reset_index()                      # remet id_maison comme colonne normale
        )
        df_conteneurs_par_maison["nb_conteneurs_total"] = (
        df_conteneurs_par_maison.drop(columns=["id_maison"]).sum(axis=1)
    )
        #print(df_conteneurs_par_maison.head(5))


        dict_commmentaires_par_id_maison = (
            df_raw_data_filtered
            .dropna(subset=["commentaires"])  # on enlève les lignes sans commentaire
            .groupby("id_maison")["commentaires"]
            .apply(lambda s: ", COM: ".join({str(x) for x in s if pd.notna(x)}))
            .to_dict()
        )

        cols=df_raw_data_filtered.columns
        print(f'COLONNES:\n {cols}')

        col_regroupt=['id_maison', 'date_visite', 'code_agent', 'x_visite', 'y_visite',  'code_insee', 'type_enquete','visite_VF']
        print('CONSERVER COLONNES:')
        print(col_regroupt)

        df_champs_generiques_par_id_maison = (
        df_raw_data_filtered[col_regroupt]
        .groupby("id_maison", as_index=False)
        .first()   # garde la "première" ligne trouvée pour chaque id_maison
        )

        #
        print('')
        print(f'nb_id_maisons_generiques: {len(df_champs_generiques_par_id_maison)}')
        print(f'nb_id_maisons_conteneurs: {len(df_conteneurs_par_maison)}')
        print(f'nb_id_maisons_commentaires_conteneurs: {len(dict_commmentaires_par_id_maison)}')
        
        df_conteneurs_par_maison["commentaires"] = df_conteneurs_par_maison["id_maison"].map(dict_commmentaires_par_id_maison).fillna("")

        df_maisons = df_champs_generiques_par_id_maison.merge(
        df_conteneurs_par_maison,
        on="id_maison",
        how="inner"
        )

        print(df_maisons.head(3))

        print(df_maisons[df_maisons['nb_conteneurs_total']==0])
        print(df_maisons.describe(include='all'))
        print(df_maisons.info())

        #CONVERSION en GEODATAFRAME
        gpd_maisons=gpd.GeoDataFrame(df_maisons, geometry=gpd.points_from_xy(df_maisons['x_visite'],df_maisons['y_visite']),crs='EPSG:2154')


        #CLUSTERISATION des POINTS
        def cluster_one_group(gdf_group, eps=100.0):
            gdf_group = gdf_group.copy()
            
            # Coordonnées en mètres
            coords = np.column_stack([
                gdf_group.geometry.x.values,
                gdf_group.geometry.y.values
            ])
            
            # DBSCAN avec min_samples=1 pour garder aussi les points isolés
            labels = DBSCAN(
                eps=eps,
                min_samples=1,
                metric='euclidean'
            ).fit_predict(coords)
            nom_col=f'cluster_{int(eps)}m'

            gdf_group[nom_col] = labels
            return gdf_group
        
        
        gdf_clustered = (
            gpd_maisons
            .groupby(["code_agent", "date_visite"], group_keys=False)
            .apply(cluster_one_group, eps=distance_max_pour_agreg_en_m)
            .reset_index(drop=True)
            )
        print(len(gdf_clustered))
        print(gdf_clustered.head(1))

        if EXPORT_TO_BD_VF:
            #EXPORT vers BD
            obj_data_access_services.export_gdf_to_postgis(gdf_clustered,tablename=nom_table_enquetes_agregees,schema=schema)
            print(f"Données 'df_maisons' exportées vers base postGis->{nom_table_enquetes_agregees}")
