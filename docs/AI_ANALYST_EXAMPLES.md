# AI Analyst API Examples

This document provides example curl requests and responses for the AI Analyst endpoint.

## Endpoint

```
POST /datasets/{dataset_id}/analyst
```

## Authentication

The endpoint supports two authentication methods:
1. **JWT Token** (preferred): Include `Authorization: Bearer <token>` header
2. **Query Parameter** (legacy): Include `?user_id=<user_id>` query parameter

## Request Format

```json
{
  "question": "string",
  "context": {
    "selected_columns": {
      "x": "column_name",
      "y": "column_name",
      "group": "column_name",
      "time": "column_name",
      "measure": "column_name"
    },
    "filters": {},
    "preferred_test": "string",
    "tone": "executive|teaching|technical",
    "detail_level": "short|medium|deep",
    "visuals": true,
    "allow_transform_plan": true
  }
}
```

---

## Example 1: Two-Group Comparison (T-Test)

### Request

```bash
curl -X POST "http://localhost:8000/datasets/abc123-def456/analyst?user_id=user789" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is there a significant difference in salary between male and female employees?",
    "context": {
      "selected_columns": {
        "y": "salary",
        "group": "gender"
      },
      "tone": "executive",
      "detail_level": "medium",
      "visuals": true
    }
  }'
```

### Response

```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "ok",
  "chosen_method": {
    "test_name": "Two-Sample T-Test",
    "analysis_slug": "two-sample-t-test",
    "why_this_test": [
      "Question asks about comparison between groups",
      "Two groups identified (male, female)",
      "Comparing numeric outcome (salary)"
    ],
    "assumptions": [
      {
        "name": "normality",
        "status": "pass",
        "evidence": "Shapiro-Wilk p > 0.05 for both groups"
      },
      {
        "name": "equal_variance",
        "status": "pass",
        "evidence": "Levene's test p = 0.23"
      }
    ],
    "alternatives_considered": [
      {
        "test": "mann-whitney-u",
        "why_not": "Data appears normally distributed"
      },
      {
        "test": "welch-t-test",
        "why_not": "Equal variances assumption holds"
      }
    ]
  },
  "data_prep": {
    "issues": [
      {
        "severity": "low",
        "column": "salary",
        "description": "Low missing rate: 2.3% of values are missing."
      }
    ],
    "suggested_fixes": [
      {
        "op": "fill_nulls",
        "args": {"column": "salary", "strategy": "median"}
      }
    ]
  },
  "transform_plan": {
    "pipeline_steps": [],
    "notes": []
  },
  "results": {
    "cached": false,
    "raw": {
      "t_statistic": 3.45,
      "p_value": 0.0012,
      "df": 198,
      "n": 200,
      "mean_diff": 8500,
      "effect_size": 0.49
    },
    "key_numbers": {
      "p_value": 0.0012,
      "effect_size": 0.49,
      "n": 200,
      "statistic": 3.45,
      "df": 198,
      "mean_diff": 8500,
      "ci": [3200, 13800]
    },
    "interpretation": {
      "plain_english": "Male employees earn significantly more than female employees. The difference of $8,500 is statistically significant (p = 0.001) with a medium effect size.",
      "statistical": "Two-Sample T-Test: t(198) = 3.45, p = 0.001, Cohen's d = 0.49 (medium effect)",
      "business_meaning": "The salary gap between genders is unlikely due to chance alone. This represents a meaningful difference that may warrant investigation into pay equity practices.",
      "decision_guidance": [
        "Result is statistically significant at α=0.05",
        "Medium effect size suggests meaningful practical impact",
        "Consider investigating root causes of the disparity"
      ],
      "risks_and_caveats": [
        "Correlation does not imply causation",
        "Other factors (experience, role, tenure) may explain the difference",
        "Assumes approximately normal distributions"
      ]
    }
  },
  "visuals": {
    "charts": [
      {
        "id": "box1",
        "title": "Boxplot: salary by gender",
        "type": "boxplot",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 400,
          "height": 300,
          "title": "Distribution of salary by gender",
          "mark": {"type": "boxplot", "extent": 1.5},
          "encoding": {
            "x": {"field": "gender", "type": "nominal"},
            "y": {"field": "salary", "type": "quantitative"},
            "color": {"field": "gender", "type": "nominal", "legend": null}
          }
        },
        "insight": "Compare the medians and spread between groups. Non-overlapping boxes suggest a meaningful difference."
      },
      {
        "id": "hist1",
        "title": "Histogram: salary by gender",
        "type": "histogram",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 400,
          "height": 300,
          "title": "Distribution of salary by gender",
          "mark": {"type": "bar", "opacity": 0.7},
          "encoding": {
            "x": {"field": "salary", "type": "quantitative", "bin": {"maxbins": 20}},
            "y": {"aggregate": "count"},
            "color": {"field": "gender", "type": "nominal"}
          }
        },
        "insight": "Overlapping distributions indicate similarity; separated distributions suggest a real difference."
      }
    ]
  },
  "next_steps": [
    "Investigate factors contributing to the salary difference",
    "Control for confounding variables (experience, role level)",
    "Consider effect size for practical significance",
    "Review company pay equity policies"
  ],
  "errors": []
}
```

---

## Example 2: Multi-Group Comparison (ANOVA)

### Request

```bash
curl -X POST "http://localhost:8000/datasets/abc123/analyst" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -d '{
    "question": "Do customer satisfaction scores differ across regions?",
    "context": {
      "selected_columns": {
        "y": "satisfaction_score",
        "group": "region"
      },
      "tone": "teaching",
      "visuals": true
    }
  }'
```

### Response

```json
{
  "analysis_id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "status": "ok",
  "chosen_method": {
    "test_name": "One-Way ANOVA",
    "analysis_slug": "anova-one-way",
    "why_this_test": [
      "Question asks about comparison among groups",
      "More than 2 groups detected (4 regions)",
      "Numeric outcome variable (satisfaction_score)"
    ],
    "assumptions": [
      {
        "name": "normality",
        "status": "pass",
        "evidence": "Residuals approximately normal"
      },
      {
        "name": "homogeneity_of_variance",
        "status": "unknown",
        "evidence": "Levene's test not performed"
      }
    ],
    "alternatives_considered": [
      {
        "test": "kruskal-wallis",
        "why_not": "Data appears normally distributed"
      },
      {
        "test": "two-sample-t-test",
        "why_not": "More than 2 groups present"
      }
    ]
  },
  "data_prep": {
    "issues": [],
    "suggested_fixes": []
  },
  "transform_plan": {
    "pipeline_steps": [],
    "notes": []
  },
  "results": {
    "cached": false,
    "raw": {
      "f_statistic": 8.72,
      "p_value": 0.00002,
      "df_between": 3,
      "df_within": 496,
      "eta_squared": 0.05
    },
    "key_numbers": {
      "p_value": 0.00002,
      "effect_size": 0.05,
      "n": 500,
      "statistic": 8.72,
      "df": [3, 496]
    },
    "interpretation": {
      "plain_english": "At least one region differs significantly from the others in customer satisfaction. The ANOVA found statistically significant differences among the four regions.",
      "statistical": "One-Way ANOVA: F(3, 496) = 8.72, p < 0.001, η² = 0.05 (small effect)",
      "business_meaning": "Regional differences in satisfaction are real, not random variation. Some regions are performing better than others in customer satisfaction.",
      "decision_guidance": [
        "Run post-hoc tests to identify which specific regions differ",
        "Effect size is small - differences exist but may be operationally modest",
        "Focus improvement efforts on underperforming regions"
      ],
      "risks_and_caveats": [
        "ANOVA only tells us groups differ, not which ones",
        "Post-hoc tests needed to identify specific differences",
        "Assumes homogeneity of variances across groups"
      ]
    }
  },
  "visuals": {
    "charts": [
      {
        "id": "box2",
        "title": "Boxplot: satisfaction_score by region",
        "type": "boxplot",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 400,
          "height": 300,
          "mark": {"type": "boxplot"},
          "encoding": {
            "x": {"field": "region", "type": "nominal"},
            "y": {"field": "satisfaction_score", "type": "quantitative"},
            "color": {"field": "region", "type": "nominal"}
          }
        },
        "insight": "Look for groups with clearly different medians."
      }
    ]
  },
  "next_steps": [
    "Run post-hoc tests (e.g., Tukey HSD) to identify which specific regions differ",
    "Investigate best practices from high-performing regions",
    "Address issues in underperforming regions"
  ],
  "errors": []
}
```

---

## Example 3: Chi-Square Test of Independence

### Request

```bash
curl -X POST "http://localhost:8000/datasets/abc123/analyst?user_id=user789" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is there an association between product category and purchase channel?",
    "context": {
      "selected_columns": {
        "x": "product_category",
        "y": "purchase_channel"
      },
      "preferred_test": "chi-square-test",
      "tone": "technical"
    }
  }'
```

### Response

```json
{
  "analysis_id": "c3d4e5f6-a7b8-9012-cdef-345678901234",
  "status": "ok",
  "chosen_method": {
    "test_name": "Chi-Square Test of Independence",
    "analysis_slug": "chi-square-test",
    "why_this_test": [
      "User specified preferred test",
      "Two categorical variables",
      "Testing independence/association"
    ],
    "assumptions": [
      {
        "name": "expected_frequency",
        "status": "pass",
        "evidence": "All expected cell frequencies >= 5"
      }
    ],
    "alternatives_considered": [
      {
        "test": "fisher-exact-test",
        "why_not": "Sample size sufficient for chi-square"
      }
    ]
  },
  "data_prep": {
    "issues": [],
    "suggested_fixes": []
  },
  "transform_plan": {
    "pipeline_steps": [],
    "notes": []
  },
  "results": {
    "cached": false,
    "raw": {
      "chi2": 45.6,
      "p_value": 0.00001,
      "df": 6,
      "cramers_v": 0.21
    },
    "key_numbers": {
      "p_value": 0.00001,
      "effect_size": 0.21,
      "chi_square": 45.6,
      "df": 6
    },
    "interpretation": {
      "plain_english": "There is a statistically significant association between product category and purchase channel. Certain products are more likely to be purchased through specific channels.",
      "statistical": "χ²(6) = 45.6, p < 0.001, Cramér's V = 0.21 (small-to-medium effect)",
      "business_meaning": "Product category and purchase channel are related. This relationship could inform targeted marketing strategies and channel optimization.",
      "decision_guidance": [
        "Strong statistical evidence of association",
        "Effect size suggests moderate practical relationship",
        "Examine the contingency table to understand the pattern"
      ],
      "risks_and_caveats": [
        "Chi-square does not indicate direction or strength of specific associations",
        "Does not establish causation",
        "May be influenced by sample size"
      ]
    }
  },
  "visuals": {
    "charts": [
      {
        "id": "stacked1",
        "title": "Stacked Bar: purchase_channel by product_category",
        "type": "stacked_bar",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 400,
          "height": 300,
          "mark": "bar",
          "encoding": {
            "x": {"field": "product_category", "type": "nominal"},
            "y": {"aggregate": "count"},
            "color": {"field": "purchase_channel", "type": "nominal"}
          }
        },
        "insight": "Uneven proportions across categories suggest an association."
      },
      {
        "id": "heatmap1",
        "title": "Heatmap: product_category × purchase_channel",
        "type": "heatmap",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 400,
          "height": 300,
          "mark": "rect",
          "encoding": {
            "x": {"field": "product_category", "type": "nominal"},
            "y": {"field": "purchase_channel", "type": "nominal"},
            "color": {"aggregate": "count", "type": "quantitative"}
          }
        },
        "insight": "Darker cells indicate higher frequencies."
      }
    ]
  },
  "next_steps": [
    "Examine specific category-channel combinations",
    "Use insights for targeted marketing",
    "Consider A/B testing channel strategies by product"
  ],
  "errors": []
}
```

---

## Example 4: Correlation Analysis

### Request

```bash
curl -X POST "http://localhost:8000/datasets/abc123/analyst?user_id=user789" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is there a relationship between advertising spend and sales revenue?",
    "context": {
      "selected_columns": {
        "x": "ad_spend",
        "y": "revenue"
      },
      "detail_level": "deep"
    }
  }'
```

### Response

```json
{
  "analysis_id": "d4e5f6a7-b8c9-0123-def0-456789012345",
  "status": "ok",
  "chosen_method": {
    "test_name": "Pearson Correlation",
    "analysis_slug": "pearson-correlation",
    "why_this_test": [
      "Question asks about relationship between variables",
      "Both variables are numeric",
      "Linear relationship assumed"
    ],
    "assumptions": [
      {
        "name": "linearity",
        "status": "unknown",
        "evidence": "Visual inspection recommended"
      }
    ],
    "alternatives_considered": [
      {
        "test": "spearman-correlation",
        "why_not": "Pearson preferred for linear relationships"
      },
      {
        "test": "simple-linear-regression",
        "why_not": "Correlation better for relationship strength, not prediction"
      }
    ]
  },
  "data_prep": {
    "issues": [],
    "suggested_fixes": []
  },
  "transform_plan": {
    "pipeline_steps": [],
    "notes": []
  },
  "results": {
    "cached": false,
    "raw": {
      "correlation": 0.78,
      "p_value": 0.00001,
      "n": 150
    },
    "key_numbers": {
      "p_value": 0.00001,
      "correlation": 0.78,
      "n": 150,
      "r_squared": 0.61
    },
    "interpretation": {
      "plain_english": "There is a strong positive correlation (r = 0.78) between advertising spend and sales revenue. Higher ad spend is associated with higher revenue.",
      "statistical": "Pearson r = 0.78, p < 0.001, n = 150. The correlation explains 61% of variance (r² = 0.61).",
      "business_meaning": "Advertising investment shows a strong positive relationship with revenue. Each dollar spent on advertising is associated with increased sales.",
      "decision_guidance": [
        "Strong correlation suggests advertising is effective",
        "61% shared variance indicates substantial relationship",
        "Consider regression analysis for prediction and optimization"
      ],
      "risks_and_caveats": [
        "Correlation does not imply causation",
        "Other factors may drive both variables",
        "Pearson only detects linear relationships",
        "Relationship may not hold at extreme values"
      ]
    }
  },
  "visuals": {
    "charts": [
      {
        "id": "scatter1",
        "title": "Scatter: ad_spend vs revenue",
        "type": "scatter",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 400,
          "height": 300,
          "layer": [
            {
              "mark": {"type": "circle", "opacity": 0.6},
              "encoding": {
                "x": {"field": "ad_spend", "type": "quantitative"},
                "y": {"field": "revenue", "type": "quantitative"}
              }
            },
            {
              "mark": {"type": "line", "color": "firebrick"},
              "transform": [{"regression": "revenue", "on": "ad_spend"}],
              "encoding": {
                "x": {"field": "ad_spend", "type": "quantitative"},
                "y": {"field": "revenue", "type": "quantitative"}
              }
            }
          ]
        },
        "insight": "r = 0.78 indicates a strong positive relationship."
      }
    ]
  },
  "next_steps": [
    "Consider regression analysis if prediction is the goal",
    "Check for non-linear relationships using scatter plot",
    "Investigate potential confounding variables",
    "Validate relationship holds across different segments"
  ],
  "errors": []
}
```

---

## Example 5: Time Trend Analysis

### Request

```bash
curl -X POST "http://localhost:8000/datasets/abc123/analyst?user_id=user789" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the trend of monthly sales over time?",
    "context": {
      "selected_columns": {
        "time": "month_date",
        "y": "monthly_sales"
      },
      "tone": "executive"
    }
  }'
```

### Response

```json
{
  "analysis_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
  "status": "ok",
  "chosen_method": {
    "test_name": "Moving Average",
    "analysis_slug": "moving-average",
    "why_this_test": [
      "Question mentions trend",
      "Datetime column available",
      "Time series analysis appropriate"
    ],
    "assumptions": [],
    "alternatives_considered": []
  },
  "data_prep": {
    "issues": [],
    "suggested_fixes": []
  },
  "transform_plan": {
    "pipeline_steps": [
      {
        "op": "sort_rows",
        "args": {"column": "month_date", "ascending": true}
      }
    ],
    "notes": ["Sorting by month_date for time series analysis"]
  },
  "results": {
    "cached": false,
    "raw": {
      "moving_average": [null, null, 102500, 105000, 108000, 112000],
      "trend_direction": "increasing",
      "trend_strength": 0.85
    },
    "key_numbers": {
      "n": 24
    },
    "interpretation": {
      "plain_english": "Monthly sales show a clear upward trend over the analysis period. The smoothed trend line reveals consistent growth despite month-to-month fluctuations.",
      "statistical": "3-month moving average shows positive trend with 85% consistency.",
      "business_meaning": "Sales are growing consistently. The business is on a positive trajectory, though seasonal variations exist.",
      "decision_guidance": [
        "Trend appears sustainable",
        "Plan for capacity to meet growing demand",
        "Monitor for any trend reversals"
      ],
      "risks_and_caveats": [
        "Past trends don't guarantee future performance",
        "External factors could disrupt the trend",
        "Seasonality should be considered separately"
      ]
    }
  },
  "visuals": {
    "charts": [
      {
        "id": "line1",
        "title": "Time Series: monthly_sales",
        "type": "line",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 600,
          "height": 300,
          "mark": {"type": "line", "point": true},
          "encoding": {
            "x": {"field": "month_date", "type": "temporal"},
            "y": {"field": "monthly_sales", "type": "quantitative"}
          }
        },
        "insight": "Look for trends, seasonality, and anomalies."
      },
      {
        "id": "run1",
        "title": "Run Chart with Median",
        "type": "run_chart",
        "spec": {
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "width": 600,
          "height": 300,
          "layer": [
            {
              "mark": {"type": "line", "point": true},
              "encoding": {
                "x": {"field": "month_date", "type": "temporal"},
                "y": {"field": "monthly_sales", "type": "quantitative"}
              }
            },
            {
              "mark": {"type": "rule", "color": "orange", "strokeDash": [4, 4]},
              "encoding": {
                "y": {"field": "monthly_sales", "aggregate": "median"}
              }
            }
          ]
        },
        "insight": "Long runs above median suggest upward trend."
      }
    ]
  },
  "next_steps": [
    "Investigate seasonality patterns",
    "Build forecasting model for planning",
    "Analyze drivers of the growth trend"
  ],
  "errors": []
}
```

---

## Error Responses

### Missing Required Information

```json
{
  "analysis_id": "f6a7b8c9-d0e1-2345-f012-678901234567",
  "status": "needs_info",
  "chosen_method": null,
  "data_prep": {
    "issues": [],
    "suggested_fixes": []
  },
  "transform_plan": {
    "pipeline_steps": [],
    "notes": []
  },
  "results": {
    "cached": false,
    "raw": {},
    "key_numbers": {},
    "interpretation": {}
  },
  "visuals": {
    "charts": []
  },
  "next_steps": [],
  "errors": [],
  "missing_info": [
    {
      "field": "group_column",
      "description": "Categorical column defining groups",
      "suggestions": ["department", "region", "category"]
    }
  ]
}
```

### Analysis Error

```json
{
  "analysis_id": "a7b8c9d0-e1f2-3456-0123-789012345678",
  "status": "error",
  "chosen_method": null,
  "data_prep": {
    "issues": [],
    "suggested_fixes": []
  },
  "transform_plan": {
    "pipeline_steps": [],
    "notes": []
  },
  "results": {
    "cached": false,
    "raw": {},
    "key_numbers": {},
    "interpretation": {}
  },
  "visuals": {
    "charts": []
  },
  "next_steps": [],
  "errors": [
    "Column 'nonexistent_column' not found in dataset"
  ]
}
```

---

## Available Tests Endpoint

Get suggested tests for a dataset:

```bash
curl -X GET "http://localhost:8000/datasets/abc123/analyst/available-tests?user_id=user789"
```

### Response

```json
{
  "dataset_id": "abc123",
  "column_summary": {
    "numeric": ["salary", "age", "experience"],
    "categorical": ["gender", "department", "region"],
    "datetime": ["hire_date"]
  },
  "suggested_tests": [
    {
      "test": "two-sample-t-test",
      "name": "Two-Sample T-Test",
      "description": "Compare means between two groups",
      "requires": {"numeric": 1, "categorical": 1},
      "suggested_columns": {
        "measure_column": "salary",
        "group_column": "gender"
      }
    },
    {
      "test": "pearson-correlation",
      "name": "Pearson Correlation",
      "description": "Measure linear relationship between two numeric variables",
      "requires": {"numeric": 2},
      "suggested_columns": {
        "x": "salary",
        "y": "age"
      }
    }
  ]
}
```
