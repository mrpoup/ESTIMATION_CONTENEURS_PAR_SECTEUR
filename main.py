import pandas as pd
from data_access.data_access_services import Data_access_services
from services import fichier_enquetes_to_agreg_sites_services as agreg_services
from services import prepa_data_services
from services import tests_prealables
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
        # X = artifacts.X.copy()

        # drop_cols = [c for c in X.columns
        #             if (c.startswith("surf_") and ("_b10_m" in c or "_b50_m" in c)
        #                 and not c.startswith("surf_buffer_m2_"))]

        # X = X.drop(columns=drop_cols)

        # Build Model C feature set (Config A)
        X_full = artifacts.X.copy()

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
        ratio_cols = [c for c in X_full.columns if c.startswith("ratio_")]
        keep_cols += ratio_cols

        # Final X for model C
        X_C = X_full[keep_cols].copy()

        # Sanity checks (minimal)
        assert X_C.isna().sum().sum() == 0, "NaNs found in X_C"
        assert np.isfinite(X_C.to_numpy()).all(), "Non-finite values found in X_C"

        print("X_C shape:", X_C.shape)
        print("Number of ratio cols:", len(ratio_cols))


        #__________________________________________
        targets="contenant enterré", "grand contenant", "petit contenant"
        target = "petit contenant"
        #__________________________________________


        TEST_BASES_LINES_VF=False
        if TEST_BASES_LINES_VF:
            tests_prealables.set_test_prealables(target,X,Y)

        
        y = artifacts.Y[target]
        baseline_B_features = [
            "log1p_surf_batiment_source_m2",
            "hauteur_corrigee_m",
            "log1p_volume_batiment",
            'log1p_surf_buffer_m2_b10_m', 'log1p_surf_buffer_m2_b50_m'
        ]

        TEST_AGGREG_ABC_VF=False
        if TEST_AGGREG_ABC_VF:
            agg_abc = simul_agreg_test.cv_aggregated_protocol_ABC(
                X_B=X_C,
                X_C=X_C,
                y=y,
                baseline_B_features=baseline_B_features,
                group_sizes=(30, 60, 120),
                n_draws=1000,
            )
            summary = agg_abc.groupby(["model", "group_size"]).agg(["mean", "std"])
            print(summary)

        TEST_AGGREG_ABC_KNN_VF=False
        TEST_AGGREG_ABC_KNN_VISU_VF=False

        if  TEST_AGGREG_ABC_KNN_VF==False:
            Test_aggreg_ABC_KNN_summaries={}

            coords = gpd_filtered_features.loc[X_C.index, ["x_visite", "y_visite"]].to_numpy()

            for target in targets:        
                y = artifacts.Y[target]

                spatial_results =simul_agreg_test.cv_spatial_knn_protocol_ABC(
                    X_B=X_C,
                    X_C=X_C,
                    coords=coords,
                    y=y,
                    baseline_B_features=baseline_B_features,
                    k_values=(30, 60, 120),
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

           

            def _extract_curve(summary, model, metric, ks=(30, 60, 120)):
                """
                summary: DataFrame with index (model, group_size) and columns MultiIndex (metric, agg)
                metric: "median_rse" or "p95_rse"
                returns: y_mean, y_std as numpy arrays aligned with ks
                """
                y_mean = []
                y_std = []
                for k in ks:
                    y_mean.append(summary.loc[(model, k), (metric, "mean")])
                    y_std.append(summary.loc[(model, k), (metric, "std")])
                return np.array(y_mean, dtype=float), np.array(y_std, dtype=float)
            
            def plot_graph1_multitarget(summaries, ks=(30, 60, 120), as_percent=True):
                """
                summaries: dict {target_name: summary_df}
                """
                targets = list(summaries.keys())

                models = [
                    ("A_mean", "Baseline A (moyenne)"),
                    ("B_neg_bin", "Baseline B (binomiale négative)"),
                    ("C_lgbm_poisson", "Modèle C (surfaces + Poisson)"),
                ]

                metrics = [
                    ("median_rse", "Erreur médiane sur le total"),
                    ("p95_rse", "Erreur dans 95% des cas (p95)"),
                ]

                fig, axes = plt.subplots(
                    nrows=len(targets),
                    ncols=len(metrics),
                    figsize=(12, 3.6 * len(targets)),
                    sharex=True
                )

                if len(targets) == 1:
                    axes = np.array([axes])  # normalise

                scale = 100.0 if as_percent else 1.0
                y_label = "Erreur relative (%)" if as_percent else "Erreur relative (ratio)"

                for i, target in enumerate(targets):
                    summary = summaries[target]

                    for j, (metric, metric_title) in enumerate(metrics):
                        ax = axes[i, j]

                        for model_code, model_label in models:
                            y_mean, y_std = _extract_curve(summary, model_code, metric, ks=ks)

                            ax.plot(ks, y_mean * scale, marker="o", label=model_label)
                            # Option: bande +/- 1 std (ça aide à “voir” la stabilité)
                            ax.fill_between(
                                ks,
                                (y_mean - y_std) * scale,
                                (y_mean + y_std) * scale,
                                alpha=0.12
                            )

                        if i == 0:
                            ax.set_title(metric_title, fontsize=12)

                        if j == 0:
                            ax.set_ylabel(f"{target}\n{y_label}")

                        ax.grid(True, alpha=0.2)
                        ax.set_xticks(list(ks))
                        ax.set_xlabel("Taille du pseudo-secteur (nombre de bâtiments)")

                # Légende unique en bas
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False)
                fig.suptitle("Précision des estimations par agrégation spatiale k-NN", fontsize=14, y=0.995)
                fig.tight_layout(rect=[0, 0.06, 1, 0.97])

                plt.show()

            plot_graph1_multitarget(
            summaries=Test_aggreg_ABC_KNN_summaries,
            ks=(30, 60, 120),
            as_percent=True
            )



            def _extract_curve(summary, model, metric, ks=(30, 60, 120)):
                y_mean = []
                y_std = []
                for k in ks:
                    y_mean.append(summary.loc[(model, k), (metric, "mean")])
                    y_std.append(summary.loc[(model, k), (metric, "std")])
                return np.array(y_mean, dtype=float), np.array(y_std, dtype=float)


            def plot_graph1_ultra_decideur(
                summaries: dict,
                ks=(30, 60, 120),
                as_percent=True
            ):
                """
                Ultra-decider plot:
                - Only p95 (risk)
                - Only A_mean vs C_lgbm_poisson
                - One panel per target
                """
                targets = list(summaries.keys())

                models = [
                    ("A_mean", "Sans modèle (moyenne)"),
                    ("C_lgbm_poisson", "Avec modèle (surfaces + Poisson)"),
                ]

                metric = "p95_rse"
                title = "Risque d’erreur sur le total d’un secteur (p95)"
                subtitle = "Interprétation : dans 95% des cas, l’erreur sur le total est inférieure à ce pourcentage"

                fig, axes = plt.subplots(
                    nrows=len(targets),
                    ncols=1,
                    figsize=(10, 3.2 * len(targets)),
                    sharex=True
                )

                if len(targets) == 1:
                    axes = [axes]

                scale = 100.0 if as_percent else 1.0
                y_label = "Erreur relative (%)" if as_percent else "Erreur relative (ratio)"

                for i, target in enumerate(targets):
                    ax = axes[i]
                    summary = summaries[target]

                    for model_code, model_label in models:
                        y_mean, y_std = _extract_curve(summary, model_code, metric, ks=ks)

                        ax.plot(ks, y_mean * scale, marker="o", linewidth=2, label=model_label)
                        ax.fill_between(
                            ks,
                            (y_mean - y_std) * scale,
                            (y_mean + y_std) * scale,
                            alpha=0.15
                        )

                        # Annotations (very readable): p95 values
                        for x, yv in zip(ks, y_mean * scale):
                            ax.annotate(f"{yv:.0f}%", (x, yv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

                    ax.set_ylabel(f"{target}\n{y_label}")
                    ax.grid(True, alpha=0.2)
                    ax.set_xticks(list(ks))

                axes[-1].set_xlabel("Taille du secteur (nombre de bâtiments)")

                # Legend + titles
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

                fig.suptitle(title, fontsize=14, y=0.995)
                fig.text(0.5, 0.965, subtitle, ha="center", fontsize=10)

                fig.tight_layout(rect=[0, 0.06, 1, 0.94])
                plt.show()

            plot_graph1_ultra_decideur(
            summaries=Test_aggreg_ABC_KNN_summaries,          # dict: {target_name: summary_df}
            ks=(30, 60, 120),
            as_percent=True
            )

                


