"""
AIAnalystAgent - Main AI Analyst Orchestrator.

The central agent that:
1) Fetches dataset profile/schema from DB
2) Decides analysis/test + params (rules first, fallback reasoning)
3) Calls run_stats to execute the analysis
4) Creates explanations (LLM optional, templated fallback)
5) Requests VizAgent to produce Vega-Lite chart specs
6) Orchestrates QA validation before returning response
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    AnalystRequest,
    AnalystResponse,
    AnalystContext,
    ChosenMethod,
    DataPrepResult,
    TransformPlan,
    AnalysisResults,
    Interpretation,
    VisualsResult,
    Assumption,
    AlternativeConsidered,
    KeyNumbers,
    MissingInfo,
    DatasetInfo,
    AnalysisSelection,
)
from .dataprep_agent import DataPrepAgent
from .transform_agent import TransformAgent
from .viz_agent import VizAgent
from .qa_agent import QAAgent
from .utils import (
    json_safe,
    detect_column_role,
    infer_analysis_from_question,
    get_test_alternatives,
    get_required_columns_for_analysis,
    format_p_value,
    interpret_p_value,
    interpret_effect_size,
)


class AIAnalystAgent:
    """
    Main AI Analyst agent that orchestrates the full analysis pipeline.

    Coordinates between DataPrepAgent, TransformAgent, VizAgent, and QAAgent
    to deliver comprehensive statistical analysis with explanations.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the AI Analyst agent.

        Args:
            openai_api_key: Optional OpenAI API key for LLM-enhanced explanations
        """
        self.openai_api_key = openai_api_key
        self.dataprep_agent = DataPrepAgent()
        self.transform_agent = TransformAgent()
        self.viz_agent = VizAgent()
        self.qa_agent = QAAgent()

    async def analyze(
        self,
        request: AnalystRequest,
        dataset_info: DatasetInfo,
        run_stats_func,
        data_sample: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalystResponse:
        """
        Perform full AI-assisted analysis.

        Args:
            request: The analyst request with question and context
            dataset_info: Information about the dataset including schema and profile
            run_stats_func: Async function to call run_stats(user_id, dataset_id, analysis, params)
            data_sample: Optional sample data for visualizations

        Returns:
            Complete AnalystResponse with analysis, explanation, and visuals
        """
        analysis_id = str(uuid.uuid4())
        errors: List[str] = []

        try:
            # Step 1: Analyze data preparation needs
            data_prep = await self._analyze_data_prep(dataset_info)

            # Step 2: Select appropriate analysis
            selection = await self._select_analysis(
                request=request,
                dataset_info=dataset_info,
            )

            # Check if we have enough info to proceed
            missing_info = self._check_required_info(selection, dataset_info)
            if missing_info:
                return AnalystResponse(
                    analysis_id=analysis_id,
                    status="needs_info",
                    data_prep=data_prep,
                    missing_info=missing_info,
                    errors=errors,
                )

            # Step 3: Build transform plan if allowed
            transform_plan = TransformPlan()
            if request.context.allow_transform_plan:
                transform_plan = await self._build_transform_plan(
                    dataset_info=dataset_info,
                    data_prep=data_prep,
                    analysis_requirements={
                        "analysis_type": selection.analysis_slug,
                        "columns": list(selection.params.values()),
                    }
                )

            # Step 4: Execute statistical analysis
            raw_result, cached = await self._execute_analysis(
                run_stats_func=run_stats_func,
                user_id=dataset_info.user_id,
                dataset_id=dataset_info.dataset_id,
                selection=selection,
            )

            # Step 5: Extract key numbers
            key_numbers = self._extract_key_numbers(raw_result, selection.analysis_slug)

            # Step 6: Build interpretation/explanation
            interpretation = await self._build_interpretation(
                question=request.question,
                selection=selection,
                raw_result=raw_result,
                key_numbers=key_numbers,
                tone=request.context.tone,
                detail_level=request.context.detail_level,
            )

            # Step 7: Generate visualizations
            visuals = VisualsResult()
            if request.context.visuals:
                visuals = await self.viz_agent.generate(
                    analysis_slug=selection.analysis_slug,
                    params=selection.params,
                    results=raw_result,
                    data_sample=data_sample,
                )

            # Step 8: Build chosen method details
            chosen_method = self._build_chosen_method(
                selection=selection,
                raw_result=raw_result,
            )

            # Step 9: Build results
            results = AnalysisResults(
                cached=cached,
                raw=json_safe(raw_result),
                key_numbers=key_numbers,
                interpretation=interpretation,
            )

            # Step 10: Generate next steps
            next_steps = self._generate_next_steps(
                selection=selection,
                key_numbers=key_numbers,
                data_prep=data_prep,
            )

            # Build response
            response = AnalystResponse(
                analysis_id=analysis_id,
                status="ok",
                chosen_method=chosen_method,
                data_prep=data_prep,
                transform_plan=transform_plan,
                results=results,
                visuals=visuals,
                next_steps=next_steps,
                errors=errors,
            )

            # Step 11: QA Validation
            is_valid, validation_errors = await self.qa_agent.validate(
                response=response,
                raw_stats_result=raw_result,
            )

            if not is_valid:
                blocking_errors = [e.message for e in validation_errors if e.severity == "error"]
                response.errors.extend(blocking_errors)
                response.status = "error"

            return response

        except Exception as e:
            return AnalystResponse(
                analysis_id=analysis_id,
                status="error",
                errors=[str(e)],
            )

    async def _analyze_data_prep(self, dataset_info: DatasetInfo) -> DataPrepResult:
        """Analyze data quality issues."""
        return await self.dataprep_agent.analyze(
            profile=dataset_info.profile,
            schema=dataset_info.schema,
        )

    async def _select_analysis(
        self,
        request: AnalystRequest,
        dataset_info: DatasetInfo,
    ) -> AnalysisSelection:
        """Select the appropriate statistical analysis."""
        # Check for preferred test
        if request.context.preferred_test:
            return AnalysisSelection(
                analysis_slug=request.context.preferred_test,
                test_name=self._get_test_display_name(request.context.preferred_test),
                params=self._build_params_from_context(request.context, dataset_info),
                reasoning=["User specified preferred test"],
                confidence=1.0,
            )

        # Infer from question and data
        selected_cols = None
        if request.context.selected_columns:
            selected_cols = {
                'x': request.context.selected_columns.x,
                'y': request.context.selected_columns.y,
                'group': request.context.selected_columns.group,
                'time': request.context.selected_columns.time,
            }

        analysis_slug, params, reasoning = infer_analysis_from_question(
            question=request.question,
            columns=dataset_info.schema,
            selected_columns=selected_cols,
        )

        # Get alternatives
        alt_tuples = get_test_alternatives(analysis_slug)
        alternatives = [
            AlternativeConsidered(test=t, why_not=r)
            for t, r in alt_tuples
        ]

        return AnalysisSelection(
            analysis_slug=analysis_slug,
            test_name=self._get_test_display_name(analysis_slug),
            params=params,
            reasoning=reasoning,
            alternatives=alternatives,
            confidence=0.8 if len(reasoning) > 2 else 0.6,
        )

    def _check_required_info(
        self,
        selection: AnalysisSelection,
        dataset_info: DatasetInfo,
    ) -> List[MissingInfo]:
        """Check if required information is available."""
        missing: List[MissingInfo] = []

        required = get_required_columns_for_analysis(selection.analysis_slug)
        col_names = {c['name'] for c in dataset_info.schema}

        for param_name, description in required.items():
            if param_name not in selection.params or not selection.params[param_name]:
                # Get suggestions based on column roles
                suggestions = []
                role_hint = "numeric" if "numeric" in description.lower() else (
                    "datetime" if "datetime" in description.lower() or "time" in description.lower()
                    else "categorical"
                )
                for col in dataset_info.schema:
                    if col.get('role') == role_hint:
                        suggestions.append(col['name'])

                missing.append(MissingInfo(
                    field=param_name,
                    description=description,
                    suggestions=suggestions[:5],
                ))

        return missing

    async def _build_transform_plan(
        self,
        dataset_info: DatasetInfo,
        data_prep: DataPrepResult,
        analysis_requirements: Dict[str, Any],
    ) -> TransformPlan:
        """Build transformation pipeline plan."""
        return await self.transform_agent.plan(
            schema=dataset_info.schema,
            suggested_fixes=data_prep.suggested_fixes,
            analysis_requirements=analysis_requirements,
        )

    async def _execute_analysis(
        self,
        run_stats_func,
        user_id: str,
        dataset_id: str,
        selection: AnalysisSelection,
    ) -> Tuple[Dict[str, Any], bool]:
        """Execute the statistical analysis."""
        result, cached = await run_stats_func(
            user_id=user_id,
            dataset_id=dataset_id,
            analysis=selection.analysis_slug,
            params=selection.params,
        )
        return result, cached

    def _extract_key_numbers(
        self,
        raw_result: Dict[str, Any],
        analysis_slug: str,
    ) -> KeyNumbers:
        """Extract key statistical numbers from raw results."""
        # Handle nested results structure
        results = raw_result.get('results', raw_result)

        # Extract common values with fallbacks
        p_value = (
            results.get('p_value') or
            results.get('p-value') or
            results.get('pvalue')
        )

        effect_size = (
            results.get('effect_size') or
            results.get('cohens_d') or
            results.get('eta_squared') or
            results.get('cramers_v')
        )

        n = (
            results.get('n') or
            results.get('n_total') or
            results.get('sample_size') or
            results.get('n_observations')
        )

        statistic = (
            results.get('statistic') or
            results.get('t_statistic') or
            results.get('f_statistic') or
            results.get('chi2') or
            results.get('test_statistic')
        )

        df = results.get('df') or results.get('degrees_of_freedom')

        ci = results.get('ci') or results.get('confidence_interval')
        if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci = [float(ci[0]), float(ci[1])]
        else:
            ci = None

        r_squared = results.get('r_squared') or results.get('r2')
        correlation = results.get('correlation') or results.get('r')
        mean_diff = results.get('mean_diff') or results.get('mean_difference')
        chi_square = results.get('chi2') or results.get('chi_square')

        return KeyNumbers(
            p_value=float(p_value) if p_value is not None else None,
            effect_size=float(effect_size) if effect_size is not None else None,
            n=int(n) if n is not None else None,
            ci=ci,
            statistic=float(statistic) if statistic is not None else None,
            df=df,
            r_squared=float(r_squared) if r_squared is not None else None,
            correlation=float(correlation) if correlation is not None else None,
            mean_diff=float(mean_diff) if mean_diff is not None else None,
            chi_square=float(chi_square) if chi_square is not None else None,
        )

    async def _build_interpretation(
        self,
        question: str,
        selection: AnalysisSelection,
        raw_result: Dict[str, Any],
        key_numbers: KeyNumbers,
        tone: str,
        detail_level: str,
    ) -> Interpretation:
        """Build interpretation of results."""
        # Try LLM if available, otherwise use templates
        if self.openai_api_key:
            try:
                return await self._build_llm_interpretation(
                    question=question,
                    selection=selection,
                    raw_result=raw_result,
                    key_numbers=key_numbers,
                    tone=tone,
                    detail_level=detail_level,
                )
            except Exception:
                pass  # Fall back to templated interpretation

        return self._build_templated_interpretation(
            question=question,
            selection=selection,
            raw_result=raw_result,
            key_numbers=key_numbers,
            tone=tone,
            detail_level=detail_level,
        )

    async def _build_llm_interpretation(
        self,
        question: str,
        selection: AnalysisSelection,
        raw_result: Dict[str, Any],
        key_numbers: KeyNumbers,
        tone: str,
        detail_level: str,
    ) -> Interpretation:
        """Build interpretation using LLM (OpenAI)."""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.openai_api_key)

            # Build prompt
            prompt = f"""You are a statistical analyst explaining results to a {tone} audience.
The user asked: "{question}"

Analysis performed: {selection.test_name}
Parameters: {selection.params}

Key results (DO NOT compute these, just explain them):
- p-value: {format_p_value(key_numbers.p_value)}
- effect size: {key_numbers.effect_size}
- sample size: {key_numbers.n}
- test statistic: {key_numbers.statistic}
- R²: {key_numbers.r_squared}
- correlation: {key_numbers.correlation}

Provide:
1. Plain English explanation (1-2 sentences)
2. Statistical interpretation (1 sentence)
3. Business meaning (1-2 sentences)
4. Decision guidance (2-3 bullet points)
5. Risks and caveats (2-3 bullet points)

Detail level: {detail_level}
Tone: {tone}

IMPORTANT: Do not compute or invent any numbers. Only use the values provided above."""

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            content = response.choices[0].message.content

            # Parse the response (simplified parsing)
            lines = content.split('\n')
            plain = ""
            statistical = ""
            business = ""
            guidance = []
            risks = []

            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "plain english" in line.lower() or "1." in line:
                    current_section = "plain"
                elif "statistical" in line.lower() or "2." in line:
                    current_section = "statistical"
                elif "business" in line.lower() or "3." in line:
                    current_section = "business"
                elif "decision" in line.lower() or "guidance" in line.lower() or "4." in line:
                    current_section = "guidance"
                elif "risk" in line.lower() or "caveat" in line.lower() or "5." in line:
                    current_section = "risks"
                elif current_section == "plain":
                    plain += line + " "
                elif current_section == "statistical":
                    statistical += line + " "
                elif current_section == "business":
                    business += line + " "
                elif current_section == "guidance" and line.startswith("-"):
                    guidance.append(line[1:].strip())
                elif current_section == "risks" and line.startswith("-"):
                    risks.append(line[1:].strip())

            return Interpretation(
                plain_english=plain.strip() or self._get_plain_english(selection, key_numbers),
                statistical=statistical.strip() or self._get_statistical_summary(selection, key_numbers),
                business_meaning=business.strip() or self._get_business_meaning(selection, key_numbers),
                decision_guidance=guidance or self._get_decision_guidance(selection, key_numbers),
                risks_and_caveats=risks or self._get_risks_caveats(selection),
            )

        except Exception:
            return self._build_templated_interpretation(
                question, selection, raw_result, key_numbers, tone, detail_level
            )

    def _build_templated_interpretation(
        self,
        question: str,
        selection: AnalysisSelection,
        raw_result: Dict[str, Any],
        key_numbers: KeyNumbers,
        tone: str,
        detail_level: str,
    ) -> Interpretation:
        """Build templated interpretation without LLM."""
        return Interpretation(
            plain_english=self._get_plain_english(selection, key_numbers),
            statistical=self._get_statistical_summary(selection, key_numbers),
            business_meaning=self._get_business_meaning(selection, key_numbers),
            decision_guidance=self._get_decision_guidance(selection, key_numbers),
            risks_and_caveats=self._get_risks_caveats(selection),
        )

    def _get_plain_english(self, selection: AnalysisSelection, key_numbers: KeyNumbers) -> str:
        """Generate plain English explanation."""
        slug = selection.analysis_slug

        if slug in ['two-sample-t-test', 'ttest_2samp', 'welch-t-test']:
            sig = interpret_p_value(key_numbers.p_value)
            if key_numbers.p_value and key_numbers.p_value < 0.05:
                return f"The two groups show a statistically significant difference ({sig}). {interpret_effect_size(key_numbers.effect_size, 'cohens_d')}"
            return f"No statistically significant difference was found between the groups ({sig})."

        elif slug in ['anova-one-way', 'anova_oneway']:
            sig = interpret_p_value(key_numbers.p_value)
            if key_numbers.p_value and key_numbers.p_value < 0.05:
                return f"At least one group differs significantly from the others ({sig}). Consider post-hoc tests to identify which groups differ."
            return f"No significant differences found among the groups ({sig})."

        elif slug in ['chi-square-test', 'chi_square']:
            sig = interpret_p_value(key_numbers.p_value)
            if key_numbers.p_value and key_numbers.p_value < 0.05:
                return f"There is a statistically significant association between the variables ({sig})."
            return f"No significant association found between the variables ({sig})."

        elif slug in ['pearson-correlation', 'spearman-correlation', 'correlation']:
            r = key_numbers.correlation
            if r is not None:
                direction = "positive" if r > 0 else "negative"
                return f"There is a {direction} {interpret_effect_size(r, 'r')} between the variables."
            return "The correlation analysis has been completed."

        elif slug in ['simple-linear-regression', 'linear_regression']:
            if key_numbers.r_squared is not None:
                pct = key_numbers.r_squared * 100
                return f"The model explains {pct:.1f}% of the variance in the outcome. {interpret_effect_size(key_numbers.r_squared, 'r2')}"
            return "The regression model has been fitted to the data."

        elif slug in ['mean', 'median', 'descriptives']:
            results = selection.params.get('column', 'the variable')
            return f"Summary statistics have been computed for {results}."

        return f"The {selection.test_name} analysis has been completed."

    def _get_statistical_summary(self, selection: AnalysisSelection, key_numbers: KeyNumbers) -> str:
        """Generate statistical summary."""
        parts = []

        if key_numbers.statistic is not None:
            parts.append(f"test statistic = {key_numbers.statistic:.4f}")
        if key_numbers.df is not None:
            parts.append(f"df = {key_numbers.df}")
        if key_numbers.p_value is not None:
            parts.append(f"p = {format_p_value(key_numbers.p_value)}")
        if key_numbers.effect_size is not None:
            parts.append(f"effect size = {key_numbers.effect_size:.4f}")
        if key_numbers.n is not None:
            parts.append(f"n = {key_numbers.n}")

        if parts:
            return f"{selection.test_name}: {', '.join(parts)}."
        return f"{selection.test_name} completed."

    def _get_business_meaning(self, selection: AnalysisSelection, key_numbers: KeyNumbers) -> str:
        """Generate business meaning interpretation."""
        slug = selection.analysis_slug

        if key_numbers.p_value is not None:
            is_sig = key_numbers.p_value < 0.05

            if slug in ['two-sample-t-test', 'ttest_2samp']:
                if is_sig:
                    return "The difference between groups is unlikely due to chance alone. This may warrant changes to strategy or operations."
                return "The observed difference could easily be due to random variation. No action is recommended based on this result alone."

            elif slug in ['anova-one-way', 'anova_oneway']:
                if is_sig:
                    return "At least one group performs differently. Investigate which group(s) to focus resources on."
                return "Groups are performing similarly. Uniform strategies may be appropriate."

            elif slug in ['chi-square-test', 'chi_square']:
                if is_sig:
                    return "The variables are related. This relationship could inform segmentation or targeting decisions."
                return "The variables appear independent. Treating them separately in analysis is appropriate."

            elif slug in ['pearson-correlation', 'spearman-correlation']:
                if key_numbers.correlation is not None:
                    r = abs(key_numbers.correlation)
                    if r > 0.5:
                        return "Strong relationship detected. Changes in one variable may reliably predict changes in the other."
                    elif r > 0.3:
                        return "Moderate relationship detected. Consider this connection in planning but don't rely on it exclusively."
                    return "Weak relationship. Other factors likely have more influence."

            elif slug in ['simple-linear-regression', 'linear_regression']:
                if key_numbers.r_squared is not None:
                    if key_numbers.r_squared > 0.5:
                        return "The model has good predictive power. It can be used for forecasting with reasonable confidence."
                    elif key_numbers.r_squared > 0.25:
                        return "The model explains some variance but other factors are also important."
                    return "The model has limited predictive power. Additional variables or a different approach may be needed."

        return "Review the statistical results in context of your specific business goals and constraints."

    def _get_decision_guidance(self, selection: AnalysisSelection, key_numbers: KeyNumbers) -> List[str]:
        """Generate decision guidance."""
        guidance = []

        if key_numbers.p_value is not None:
            if key_numbers.p_value < 0.05:
                guidance.append("Result is statistically significant at α=0.05. Consider the practical significance alongside statistical significance.")
                if key_numbers.effect_size is not None:
                    if abs(key_numbers.effect_size) < 0.2:
                        guidance.append("Effect size is small. Even though significant, the practical impact may be minimal.")
                    elif abs(key_numbers.effect_size) > 0.8:
                        guidance.append("Large effect size suggests meaningful practical impact.")
            else:
                guidance.append("Result is not statistically significant. Consider whether sample size is adequate.")
                guidance.append("Absence of evidence is not evidence of absence. A larger sample might reveal effects.")

        if key_numbers.n is not None:
            if key_numbers.n < 30:
                guidance.append(f"Sample size (n={key_numbers.n}) is small. Results should be interpreted cautiously.")
            elif key_numbers.n > 1000:
                guidance.append("Large sample size. Even small effects may appear significant; focus on effect size.")

        if not guidance:
            guidance = [
                "Review assumptions before making decisions",
                "Consider replicating the analysis with new data",
                "Consult domain experts for context",
            ]

        return guidance

    def _get_risks_caveats(self, selection: AnalysisSelection) -> List[str]:
        """Generate risks and caveats."""
        caveats = []
        slug = selection.analysis_slug

        # General caveats
        caveats.append("Correlation does not imply causation")
        caveats.append("Results depend on data quality and representativeness")

        # Test-specific caveats
        if slug in ['two-sample-t-test', 'ttest_2samp']:
            caveats.append("Assumes approximately normal distributions or large samples")
            caveats.append("Sensitive to outliers")

        elif slug in ['anova-one-way', 'anova_oneway']:
            caveats.append("Assumes homogeneity of variances across groups")
            caveats.append("Post-hoc tests needed to identify specific group differences")

        elif slug in ['chi-square-test', 'chi_square']:
            caveats.append("Expected cell frequencies should be ≥5 for reliable results")
            caveats.append("Does not indicate strength or direction of association")

        elif slug in ['pearson-correlation', 'correlation']:
            caveats.append("Only detects linear relationships")
            caveats.append("Sensitive to outliers")

        elif slug in ['simple-linear-regression', 'linear_regression']:
            caveats.append("Assumes linear relationship between variables")
            caveats.append("Check residuals for normality and homoscedasticity")

        return caveats[:5]  # Limit to 5 caveats

    def _build_chosen_method(
        self,
        selection: AnalysisSelection,
        raw_result: Dict[str, Any],
    ) -> ChosenMethod:
        """Build the chosen method details."""
        # Extract assumptions from raw result if available
        assumptions = []
        raw_assumptions = raw_result.get('assumptions', {})
        if isinstance(raw_assumptions, dict):
            for name, info in raw_assumptions.items():
                if isinstance(info, dict):
                    assumptions.append(Assumption(
                        name=name,
                        status=info.get('status', 'unknown'),
                        evidence=info.get('evidence', ''),
                    ))
                else:
                    assumptions.append(Assumption(
                        name=name,
                        status='unknown',
                        evidence=str(info),
                    ))

        return ChosenMethod(
            test_name=selection.test_name,
            analysis_slug=selection.analysis_slug,
            why_this_test=selection.reasoning,
            assumptions=assumptions,
            alternatives_considered=selection.alternatives,
        )

    def _generate_next_steps(
        self,
        selection: AnalysisSelection,
        key_numbers: KeyNumbers,
        data_prep: DataPrepResult,
    ) -> List[str]:
        """Generate recommended next steps."""
        steps = []
        slug = selection.analysis_slug

        # Data quality-based steps
        high_severity = sum(1 for i in data_prep.issues if i.severity == 'high')
        if high_severity > 0:
            steps.append(f"Address {high_severity} high-severity data quality issues before finalizing conclusions")

        # Analysis-specific next steps
        if slug in ['anova-one-way', 'anova_oneway'] and key_numbers.p_value and key_numbers.p_value < 0.05:
            steps.append("Run post-hoc tests (e.g., Tukey HSD) to identify which specific groups differ")

        elif slug in ['pearson-correlation', 'correlation']:
            steps.append("Consider regression analysis if prediction is the goal")
            steps.append("Check for non-linear relationships using scatter plot")

        elif slug in ['simple-linear-regression', 'linear_regression']:
            steps.append("Examine residual plots to verify model assumptions")
            steps.append("Consider adding more predictors if R² is low")

        elif slug in ['two-sample-t-test', 'ttest_2samp']:
            steps.append("Consider effect size for practical significance")
            if key_numbers.p_value and key_numbers.p_value > 0.05:
                steps.append("Power analysis may help determine if sample size was adequate")

        # General steps
        if not steps:
            steps = [
                "Validate findings with domain experts",
                "Consider replication with independent data",
                "Document methodology for reproducibility",
            ]

        return steps[:5]

    def _build_params_from_context(
        self,
        context: AnalystContext,
        dataset_info: DatasetInfo,
    ) -> Dict[str, Any]:
        """Build analysis params from context."""
        params = {}

        if context.selected_columns:
            if context.selected_columns.x:
                params['x'] = context.selected_columns.x
            if context.selected_columns.y:
                params['y'] = context.selected_columns.y
            if context.selected_columns.group:
                params['group_column'] = context.selected_columns.group
            if context.selected_columns.time:
                params['time_column'] = context.selected_columns.time
            if context.selected_columns.measure:
                params['measure_column'] = context.selected_columns.measure

        return params

    def _get_test_display_name(self, slug: str) -> str:
        """Get human-readable test name from slug."""
        names = {
            "two-sample-t-test": "Two-Sample T-Test",
            "ttest_2samp": "Two-Sample T-Test",
            "one-sample-t-test": "One-Sample T-Test",
            "paired-t-test": "Paired T-Test",
            "welch-t-test": "Welch's T-Test",
            "anova-one-way": "One-Way ANOVA",
            "anova_oneway": "One-Way ANOVA",
            "kruskal-wallis": "Kruskal-Wallis Test",
            "chi-square-test": "Chi-Square Test of Independence",
            "chi_square": "Chi-Square Test",
            "fisher-exact-test": "Fisher's Exact Test",
            "pearson-correlation": "Pearson Correlation",
            "spearman-correlation": "Spearman Correlation",
            "kendall-tau": "Kendall's Tau",
            "correlation": "Correlation Analysis",
            "simple-linear-regression": "Simple Linear Regression",
            "linear_regression": "Linear Regression",
            "multiple-linear-regression": "Multiple Linear Regression",
            "polynomial-regression": "Polynomial Regression",
            "mean": "Mean (Arithmetic Average)",
            "median": "Median",
            "variance": "Variance",
            "descriptives": "Descriptive Statistics",
            "normality-test": "Normality Test",
            "moving-average": "Moving Average",
        }
        return names.get(slug, slug.replace("-", " ").replace("_", " ").title())
