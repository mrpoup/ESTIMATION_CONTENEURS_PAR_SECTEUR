from services import modeles_services_regression
from services import metrics_services_regression
from services import comparison_regression_models_services

def set_test_A_B(target:str,X,Y,avec_agreg:bool, baseline_B_features = [
        "log1p_surf_batiment_source_m2",
        "hauteur_corrigee_m",
        "log1p_volume_batiment",
        'log1p_surf_buffer_m2_b10_m', 'log1p_surf_buffer_m2_b50_m'
    ]
    
    ,pks=[30,60,120], p_n_draws=1000
    ):
    y_true = Y[target]

    baseline_A =modeles_services_regression.BaselineMeanPredictor().fit(y_true)
    y_pred_A = baseline_A.predict(len(y_true))

    metrics_A = metrics_services_regression.CountRegressionMetrics.compute_all(y_true, y_pred_A)

    print(f'\n{target} Metrics sur baseline mean:')
    print(metrics_A)

    #

    baseline_B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
    baseline_B.fit(X, y_true)

    y_pred_B = baseline_B.predict(X)

    metrics_B = metrics_services_regression.CountRegressionMetrics.compute_all(y_true, y_pred_B)

    print(f'\n{target} sur baseline regression binomiale negative:')
    print(metrics_B)


    cv_results = comparison_regression_models_services.cross_validate_baselines(
        X=X,
        y=Y[target],
        baseline_B_features=baseline_B_features,
    )
    # print(f'\n{target} CROSS VALIDATION RESULTS:  ')
    # print(cv_results)

    summary = (
        cv_results
        .groupby("baseline")
        .agg(["mean", "std"])
    )
    print(f'\n{target} CROSS VALIDATION SUMMARY RESULTS: ')
    print(summary)

    #SIMULATION d'AGREGATION:
    if avec_agreg:
        agg_results = comparison_regression_models_services.cv_aggregated_protocol_AB(
            X=X,
            y=Y[target],
            baseline_B_features=baseline_B_features,
            group_sizes=pks,
            n_draws=p_n_draws,
        )
        # print(f'\n{target} CROSS VALIDATION ON AGGREG. RESULTS: ')
        # print(agg_results)

        summary = (
            agg_results
            .groupby(["model", "group_size"])
            .agg(["mean", "std"])
        )

        print(f'\n{target} CROSS VALIDATION ON AGGREG. SUMMARY RESULTS: ')
        print(agg_results)
        print(summary)