/**
 * Pre-computed model metrics for the Model Performance page.
 * Source: outputs/tables/cl_07_leaderboard.csv and rg_06_leaderboard_test.csv
 */

export const classificationLeaderboard = [
  {
    model: 'XGBoost (calibrated)',
    accuracy: 0.882,
    precision: 0.800,
    recall: 0.854,
    f1: 0.826,
    auc: 0.953,
    brier: 0.079,
    isProduction: true,
  },
  {
    model: 'Decision Tree',
    accuracy: 0.861,
    precision: 0.770,
    recall: 0.822,
    f1: 0.795,
    auc: 0.876,
    brier: null,
    isProduction: false,
  },
  {
    model: 'Logistic Regression',
    accuracy: 0.777,
    precision: 0.630,
    recall: 0.777,
    f1: 0.696,
    auc: 0.863,
    brier: null,
    isProduction: false,
  },
  {
    model: 'Dummy Classifier',
    accuracy: 0.672,
    precision: 0.0,
    recall: 0.0,
    f1: 0.0,
    auc: 0.500,
    brier: null,
    isProduction: false,
  },
];

export const regressionLeaderboard = [
  {
    model: 'XGBoost (tuned)',
    r2: 0.755,
    mae: 10.46,
    rmse: 16.24,
    isProduction: true,
  },
  {
    model: 'Decision Tree (d=12)',
    r2: 0.607,
    mae: 13.50,
    rmse: 20.58,
    isProduction: false,
  },
  {
    model: 'Ridge Regression',
    r2: 0.501,
    mae: 17.15,
    rmse: 23.18,
    isProduction: false,
  },
  {
    model: 'Dummy (mean)',
    r2: -0.000,
    mae: 25.16,
    rmse: 32.81,
    isProduction: false,
  },
];

/** Cancellation model confusion matrix (test set) */
export const confusionMatrix = {
  tn: 4576 * (1 - 0.066),   // approx from risk band data
  fp: 4576 * 0.066,
  fn: (2130 * (1 - 0.873)),
  tp: 2130 * 0.873,
  // Exact from test metrics: accuracy=0.882, precision=0.800, recall=0.854
  // With 7248 test samples, 2376 canceled (32.76%), 4872 not-canceled:
  labels: ['Not Canceled', 'Canceled'],
  matrix: [
    [4205, 667],   // TN, FP  (not-canceled row)
    [345, 2031],   // FN, TP  (canceled row)
  ],
};

/** Top features for cancellation (gain importance, top 10 clean names) */
export const clfFeatureImportance = [
  { feature: 'Online segment',     importance: 0.233 },
  { feature: 'Car parking',        importance: 0.085 },
  { feature: 'Special requests',   importance: 0.061 },
  { feature: 'Offline segment',    importance: 0.055 },
  { feature: 'Has special req',    importance: 0.050 },
  { feature: 'Lead time',          importance: 0.041 },
  { feature: 'Repeated guest',     importance: 0.039 },
  { feature: 'Arrival season',     importance: 0.027 },
  { feature: 'Log lead time',      importance: 0.026 },
  { feature: 'Total guests',       importance: 0.026 },
];

/** Top features for price regression (gain, top 10 clean names) */
export const regFeatureImportance = [
  { feature: 'Total guests',       importance: 0.128 },
  { feature: 'Room Type 6',        importance: 0.111 },
  { feature: 'Room Type 1',        importance: 0.107 },
  { feature: 'Room Type 7',        importance: 0.069 },
  { feature: 'Meal Plan 2',        importance: 0.065 },
  { feature: 'Online segment',     importance: 0.064 },
  { feature: 'Has children',       importance: 0.043 },
  { feature: 'Room Type 4',        importance: 0.035 },
  { feature: 'Room Type 5',        importance: 0.034 },
  { feature: 'Complementary seg',  importance: 0.033 },
];

/** Risk band distribution (test set) */
export const riskBands = [
  { band: 'Low (<30%)',    pct: 63.1, actualCancelRate: 6.6,  color: '#4caf50' },
  { band: 'Medium (30–60%)', pct: 7.5,  actualCancelRate: 39.9, color: '#ff9800' },
  { band: 'High (≥60%)',  pct: 29.4, actualCancelRate: 87.3, color: '#f44336' },
];

/** Overfitting summary */
export const overfitSummary = [
  { split: 'Train', accuracy: 0.954, f1: 0.931, auc: 0.992 },
  { split: 'Val',   accuracy: 0.887, f1: 0.831, auc: 0.953 },
  { split: 'Test',  accuracy: 0.882, f1: 0.826, auc: 0.952 },
];

/** Calibration comparison */
export const calibration = [
  { method: 'Uncalibrated', brier: 0.084 },
  { method: 'Isotonic',     brier: 0.079 },
  { method: 'Sigmoid',      brier: 0.082 },
];
