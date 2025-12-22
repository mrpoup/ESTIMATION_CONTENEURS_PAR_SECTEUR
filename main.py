import pandas as pd
from data_access.data_access_services import Data_access_services
from services import fichier_enquetes_to_agreg_sites_services as agreg_services
from services import prepa_data_services
from services import modeles_services_regression
from services import comparison_models_regression
from services import visu
from services.classification_models import classif_baselines
from services.classification_models.adapters import RegressionToClassBaseline
from services import calcul_classes_services
from services import comparison_models_classif

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
   
    #sys.exit()
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
    #gpd_filtered_features=gpd_filtered_features[gpd_filtered_features['type_enquete'].isin(['plainte','enquete_entomo'])]
 
    print(f'\nFILTERED FEATURES:')
    print(f'->{len(gpd_filtered_features)} ligne(s)')
    print(f'->col: \n{gpd_filtered_features.columns}')

    print(gpd_filtered_features['type_enquete'].unique())
    
   


    print(gpd_filtered_features[['contenant enterré','grand contenant', 'petit contenant']].describe())

    print(gpd_filtered_features.columns)


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

    #CORRIGE HAUTEURS:
    DEFAULT_HEIGHT = 5.0

    height_effective = (
        gpd_filtered_features["hauteur"]
        .astype(float)
        .fillna(DEFAULT_HEIGHT)
    )
    gpd_filtered_features["hauteur_corrigee_m"]=height_effective

    features_col.remove('hauteur')
    features_col=features_col+["hauteur_corrigee_m"]


    TEST_ABC_VF=True
    if TEST_ABC_VF:
        service = prepa_data_services.BuildingsCountDataPreparationService(id_col="id_maison")

        dataset_obj = service.prepare_dataset(
            df=gpd_filtered_features,
            features_col=features_col,
            targets_col=targets_col,
            add_volume=True,
            make_ratios=True,
            add_log1p=True,
            target_cols_to_keep=["contenant enterré", "grand contenant", "petit contenant"],
        )

        X, Y, ids = dataset_obj.X, dataset_obj.Y, dataset_obj.ids

        print(f'->colonnes x:\n {X.columns}')
        #sys.exit()
        # print(f'->colonnes y:\n {Y.columns}')


        #On peut enlever les surfaces brutes:
        # X = artifacts.X.copy()

        # drop_cols = [c for c in X.columns
        #             if (c.startswith("surf_") and ("_b10_m" in c or "_b50_m" in c)
        #                 and not c.startswith("surf_buffer_m2_"))]

        # X = X.drop(columns=drop_cols)

        # Build Model C feature set (Config A)
        X_all_features = dataset_obj.X.copy()

        keep_cols = []

        # C1: building-scale features
        keep_cols += [
            "surf_batiment_source_m2",
            "hauteur_corrigee_m",
            "volume_batiment",
            "log1p_surf_batiment_source_m2",
            "log1p_volume_batiment",
        ]

        # C2: buffer scale features
        keep_cols += [
            "surf_buffer_m2_b10_m",
            "surf_buffer_m2_b50_m",
            "log1p_surf_buffer_m2_b10_m",
            "log1p_surf_buffer_m2_b50_m",
        ]

        # C3: composition ratios
        ratio_cols = [c for c in X_all_features.columns if c.startswith("ratio_")]
        keep_cols += ratio_cols

        # Final X for model C
        X_C = X_all_features[keep_cols].copy()

        # Sanity checks (minimal)
        assert X_C.isna().sum().sum() == 0, "NaNs found in X_C"
        assert np.isfinite(X_C.to_numpy()).all(), "Non-finite values found in X_C"

        print("X_C shape:", X_C.shape)
        print("Number of ratio cols:", len(ratio_cols))


        


        TEST_BASES_LINES_VF=False
        TEST_AGGREG_ABC_SANS_KNN_VF=False
        #
        TEST_AGGREG_ABC_KNN_VF=False
        TEST_AGGREG_ABC_KNN_VISU_VF=False
        #
        TEST_VISU_COURBES_REGRESSION_VF=False  
        #
        TEST_APPROCHE_par_CLASSES_VF=True

        if TEST_VISU_COURBES_REGRESSION_VF or TEST_AGGREG_ABC_KNN_VISU_VF:
            TEST_AGGREG_ABC_KNN_VF=True


        #__________________________________________
        targets="contenant enterré", "grand contenant", "petit contenant"
        #__________________________________________
        
        baseline_B_features = [
            "log1p_surf_batiment_source_m2",
            "hauteur_corrigee_m",
            "log1p_volume_batiment",
            'log1p_surf_buffer_m2_b10_m', 'log1p_surf_buffer_m2_b50_m'
        ]

        pks=(30, 60, 120,180,240,300)
        coords = gpd_filtered_features.loc[X_C.index, ["x_visite", "y_visite"]].to_numpy()


        if TEST_BASES_LINES_VF:
            #targets="contenant enterré", "grand contenant", "petit contenant"
            target = "contenant enterré"
            #
            comparison_models_regression.set_test_A_B(target,X,Y)

        
        
        if TEST_AGGREG_ABC_SANS_KNN_VF:
            #targets="contenant enterré", "grand contenant", "petit contenant"
            target = "contenant enterré"
            #
            y = dataset_obj.Y[target]
            agg_abc = comparison_models_regression.cv_aggregated_protocol_ABC(
                X_B=X_C,
                X_C=X_C,
                y=y,
                baseline_B_features=baseline_B_features,
                group_sizes=pks,
                n_draws=1000,
            )
            summary = agg_abc.groupby(["model", "group_size"]).agg(["mean", "std"])
            print(summary)

        

        if  TEST_AGGREG_ABC_KNN_VF:
            Test_aggreg_ABC_KNN_summaries={}

            for target in targets:        
                y = dataset_obj.Y[target]

                spatial_results =comparison_models_regression.cv_spatial_knn_protocol_ABC(
                    X_B=X_C,
                    X_C=X_C,
                    coords=coords,
                    y=y,
                    baseline_B_features=baseline_B_features,
                    k_values=pks,
                )

                summary_spatial = (
                    spatial_results
                    .groupby(["model", "group_size"])
                    .agg(["mean", "std"])
                )
               
                Test_aggreg_ABC_KNN_summaries[target]=summary_spatial

            if TEST_AGGREG_ABC_KNN_VISU_VF:
                print(target)
                print(summary_spatial)

                visu.plot_graph1_multitarget(
                summaries=Test_aggreg_ABC_KNN_summaries,
                ks=pks ,
                as_percent=True
                )

                visu.plot_graph1_ultra_decideur(
                summaries=Test_aggreg_ABC_KNN_summaries,          # dict: {target_name: summary_df}
                ks=pks,
                as_percent=True
                )

        
        if TEST_VISU_COURBES_REGRESSION_VF:
            #targets="contenant enterré", "grand contenant", "petit contenant"
            target = "contenant enterré"
            k = 60
            max_groups_per_fold=100
            #
            modelC_factory = lambda: modeles_services_regression.ModelCPoissonLGBM(params=None, random_state=42)
            #
            true_sums_C, pred_sums_C = comparison_models_regression.cv_collect_group_sums_modelC(
            X_C=X_C,
            coords=coords,
            y=y,
            modelC_factory=lambda: modeles_services_regression.ModelCPoissonLGBM(params=None, random_state=42),
            k=k,
            max_groups_per_fold=max_groups_per_fold
        )

            visu.plot_true_vs_pred_sector_sums(
                true_sums_C, pred_sums_C,
                title=f"{target} — Totaux par pseudo-secteur k={k} (CV, modèle C)"
            )


    if TEST_APPROCHE_par_CLASSES_VF:
        #targets="contenant enterré", "grand contenant", "petit contenant"
        target = "contenant enterré"

        #methodes="quantile", "thresholds", "balanced_integers"
        methode="balanced_integers"
        n_classes=10
        creer_une_classe_specifique_pour_zero_vf=True
        y=dataset_obj.Y[target]
        #
        service =calcul_classes_services.TargetBinningService()

        art = service.build_classes(y,
            spec=calcul_classes_services.BinningSpec(method=methode, n_classes=n_classes, zero_as_own_class=creer_une_classe_specifique_pour_zero_vf),
        )

        y_class = art.y_class
        bin_log = art.log
        #
        edges=bin_log['edges']
        print(edges)
        #sys.exit()

        nb_lignes=sum([v for v in bin_log["class_counts"].values() ])

        class_bounds = [
        {
            "class_id": i,
            "lower_bound": low,
            "upper_bound": high
        }
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
        ]
        #
        print('\n___________________________________')
        print(art.column_name)
        print(f'nb indiv par classe:{bin_log["class_counts"]}')
       
        print(f'nb classes: {len(bin_log["class_counts"])} vs attendu: {n_classes}')
        nb_lignes=sum([v for v in bin_log["class_counts"].values() ])
        print(f'nb objets classés: {nb_lignes}')
        print(f'edges: {edges}')
        print(f'\nlimites de classes:')
       
        for b in class_bounds:
            print(f'{b['class_id']}: {b['lower_bound']} -> {b['upper_bound']}')
        print('___________________________________')


        print(f'Colonne classe:')
        class_col_name=f'Class_{target}_{methode[0:2]}{n_classes}'
        print(class_col_name)
        #
        dataset_obj.X[class_col_name]=y_class


        #COMPARAISON des MODELES:
        regressor =modeles_services_regression.ModelCPoissonLGBM()

        binning_service =calcul_classes_services.TargetBinningService()
        binning_spec = calcul_classes_services.BinningSpec(
            method=methode,
            n_classes=n_classes,
            zero_as_own_class=False,
        )

        y_float = dataset_obj.Y["contenant enterré"]  
       

        # tmp = binning_service.build_classes(y=y_float, spec=binning_spec)
        # print(tmp.log.keys())
        # print("thresholds_finite:", tmp.log.get("thresholds_finite"))
        # print("edges_with_inf:", tmp.log.get("edges_with_inf"))

        # print("methode:", methode)
        # tmp = binning_service.build_classes(y=y_float, spec=binning_spec)
        # print({k: type(v) for k, v in tmp.log.items()})

      
       
        models = [
            
            classif_baselines.BaselineMajorityClass(),
            classif_baselines.BaselineStratifiedRandom(random_state=42),
            RegressionToClassBaseline(
                regressor=regressor,
                binning_service=binning_service,
                binning_spec=binning_spec,
                y_continuous=y_float,
            ),
        ]

    

        X_all_features = dataset_obj.X.copy()

        keep_cols = []

        # C1: building-scale features
        keep_cols += [
            "surf_batiment_source_m2",
            "hauteur_corrigee_m",
            "volume_batiment",
            "log1p_surf_batiment_source_m2",
            "log1p_volume_batiment",
        ]
        X_B = X_all_features[keep_cols].copy()

        # C2: buffer scale features
        keep_cols += [
            "surf_buffer_m2_b10_m",
            "surf_buffer_m2_b50_m",
            "log1p_surf_buffer_m2_b10_m",
            "log1p_surf_buffer_m2_b50_m",
        ]

        # C3: composition ratios
        ratio_cols = [c for c in X_all_features.columns if c.startswith("ratio_")]
        keep_cols += ratio_cols

        # Final X for model C
        X_C = X_all_features[keep_cols].copy()

        # Sanity checks (minimal)
        assert X_C.isna().sum().sum() == 0, "NaNs found in X_C"
        assert np.isfinite(X_C.to_numpy()).all(), "Non-finite values found in X_C"

        print("X_C shape:", X_C.shape)
        print("Number of ratio cols:", len(ratio_cols))


        y_cls=X_all_features[class_col_name]
   

        #TEST X_B
        TEST_CLASSIF_X_B_FV=False
        if TEST_CLASSIF_X_B_FV:
            classModelComparisonService=comparison_models_classif.ClassModelComparisonService()
            res_XB = classModelComparisonService.compare(
            X=X_B,
            y_class=y_cls,
            models=models,
            )
            print('|nX_B:')
            print(res_XB.summary)

        #TEST X_C
        TEST_CLASSIF_X_C_FV=True
        if TEST_CLASSIF_X_C_FV:
            classModelComparisonService=comparison_models_classif.ClassModelComparisonService()
            res_XC = classModelComparisonService.compare(
            X=X_C,
            y_class=y_cls,
            models=models,
            )
            print('\nX_C:')
            print(res_XC.summary)

       





