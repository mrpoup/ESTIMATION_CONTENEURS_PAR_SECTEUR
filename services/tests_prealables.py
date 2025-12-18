from services import modeles_services
from services import metrics_services
from services import cross_validation_services
from services import simul_agreg_test


def set_test_prealables(target:str,X,Y):
    y_true = Y[target]

    baseline_A =modeles_services.BaselineMeanPredictor().fit(y_true)
    y_pred_A = baseline_A.predict(len(y_true))

    metrics_A = metrics_services.CountRegressionMetrics.compute_all(y_true, y_pred_A)

    print(f'\n{target} Metrics sur baseline mean:')
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

    print(f'\n{target} sur baseline regression binomiale negative:')
    print(metrics_B)


    cv_results = cross_validation_services.cross_validate_baselines(
        X=X,
        y=Y[target],
        baseline_B_features=baseline_B_features,
    )
    print(f'\n{target} CROSS VALIDATION RESULTS:  ')
    print(cv_results)

    summary = (
        cv_results
        .groupby("baseline")
        .agg(["mean", "std"])
    )
    print(f'\n{target} CROSS VALIDATION SUMMARY RESULTS: ')
    print(summary)

    #SIMULATION d'AGREGATION:
    groupes=(30, 60, 120)
    agg_results = simul_agreg_test.cv_aggregated_protocol_AB(
        X=X,
        y=Y[target],
        baseline_B_features=baseline_B_features,
        group_sizes=groupes,
        n_draws=1000,
    )
    print(f'\n{target} CROSS VALIDATION ON AGGREG. RESULTS: ')
    print(agg_results)

    summary = (
        agg_results
        .groupby(["model", "group_size"])
        .agg(["mean", "std"])
    )

    print(f'\n{target} CROSS VALIDATION ON AGGREG. SUMMARY RESULTS: ')
    print(agg_results)
    print(summary)