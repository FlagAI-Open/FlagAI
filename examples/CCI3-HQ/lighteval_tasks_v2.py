# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

Do note that we ran the evals with `max_samples=1000` to speed up large evals.
Most custom prompt changes were in an attempt to improve signal for small models in general.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Example usage (lighteval_tasks.py is the path to this file):
===================
accelerate launch --num_processes=1 lighteval/run_evals_accelerate.py --model_args="pretrained=HuggingFaceFW/ablation-model-fineweb-edu" \
    --custom_tasks "lighteval_tasks.py" --output_dir [OUTPUTPATH] --max_samples 1000 \ 
    --tasks "custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|mmlu:abstract_algebra|0|1,custom|mmlu:anatomy|0|1,custom|mmlu:astronomy|0|1,custom|mmlu:business_ethics|0|1,custom|mmlu:clinical_knowledge|0|1,custom|mmlu:college_biology|0|1,custom|mmlu:college_chemistry|0|1,custom|mmlu:college_computer_science|0|1,custom|mmlu:college_mathematics|0|1,custom|mmlu:college_medicine|0|1,custom|mmlu:college_physics|0|1,custom|mmlu:computer_security|0|1,custom|mmlu:conceptual_physics|0|1,custom|mmlu:econometrics|0|1,custom|mmlu:electrical_engineering|0|1,custom|mmlu:elementary_mathematics|0|1,custom|mmlu:formal_logic|0|1,custom|mmlu:global_facts|0|1,custom|mmlu:high_school_biology|0|1,custom|mmlu:high_school_chemistry|0|1,custom|mmlu:high_school_computer_science|0|1,custom|mmlu:high_school_european_history|0|1,custom|mmlu:high_school_geography|0|1,custom|mmlu:high_school_government_and_politics|0|1,custom|mmlu:high_school_macroeconomics|0|1,custom|mmlu:high_school_mathematics|0|1,custom|mmlu:high_school_microeconomics|0|1,custom|mmlu:high_school_physics|0|1,custom|mmlu:high_school_psychology|0|1,custom|mmlu:high_school_statistics|0|1,custom|mmlu:high_school_us_history|0|1,custom|mmlu:high_school_world_history|0|1,custom|mmlu:human_aging|0|1,custom|mmlu:human_sexuality|0|1,custom|mmlu:international_law|0|1,custom|mmlu:jurisprudence|0|1,custom|mmlu:logical_fallacies|0|1,custom|mmlu:machine_learning|0|1,custom|mmlu:management|0|1,custom|mmlu:marketing|0|1,custom|mmlu:medical_genetics|0|1,custom|mmlu:miscellaneous|0|1,custom|mmlu:moral_disputes|0|1,custom|mmlu:moral_scenarios|0|1,custom|mmlu:nutrition|0|1,custom|mmlu:philosophy|0|1,custom|mmlu:prehistory|0|1,custom|mmlu:professional_accounting|0|1,custom|mmlu:professional_law|0|1,custom|mmlu:professional_medicine|0|1,custom|mmlu:professional_psychology|0|1,custom|mmlu:public_relations|0|1,custom|mmlu:security_studies|0|1,custom|mmlu:sociology|0|1,custom|mmlu:us_foreign_policy|0|1,custom|mmlu:virology|0|1,custom|mmlu:world_religions|0|1"
===================
custom|cmmlu:agronomy|0|1,custom|cmmlu:anatomy|0|1,custom|cmmlu:ancient_chinese|0|1,custom|cmmlu:arts|0|1,custom|cmmlu:astronomy|0|1,custom|cmmlu:business_ethics|0|1,custom|cmmlu:chinese_civil_service_exam|0|1,custom|cmmlu:chinese_driving_rule|0|1,custom|cmmlu:chinese_food_culture|0|1,custom|cmmlu:chinese_foreign_policy|0|1,custom|cmmlu:chinese_history|0|1,custom|cmmlu:chinese_literature|0|1,custom|cmmlu:chinese_teacher_qualification|0|1,custom|cmmlu:clinical_knowledge|0|1,custom|cmmlu:college_actuarial_science|0|1,custom|cmmlu:college_education|0|1,custom|cmmlu:college_engineering_hydrology|0|1,custom|cmmlu:college_law|0|1,custom|cmmlu:college_mathematics|0|1,custom|cmmlu:college_medical_statistics|0|1,custom|cmmlu:college_medicine|0|1,custom|cmmlu:computer_science|0|1,custom|cmmlu:computer_security|0|1,custom|cmmlu:conceptual_physics|0|1,custom|cmmlu:construction_project_management|0|1,custom|cmmlu:economics|0|1,custom|cmmlu:education|0|1,custom|cmmlu:electrical_engineering|0|1,custom|cmmlu:elementary_chinese|0|1,custom|cmmlu:elementary_commonsense|0|1,custom|cmmlu:elementary_information_and_technology|0|1,custom|cmmlu:elementary_mathematics|0|1,custom|cmmlu:ethnology|0|1,custom|cmmlu:food_science|0|1,custom|cmmlu:genetics|0|1,custom|cmmlu:global_facts|0|1,custom|cmmlu:high_school_biology|0|1,custom|cmmlu:high_school_chemistry|0|1,custom|cmmlu:high_school_geography|0|1,custom|cmmlu:high_school_mathematics|0|1,custom|cmmlu:high_school_physics|0|1,custom|cmmlu:high_school_politics|0|1,custom|cmmlu:human_sexuality|0|1,custom|cmmlu:international_law|0|1,custom|cmmlu:journalism|0|1,custom|cmmlu:jurisprudence|0|1,custom|cmmlu:legal_and_moral_basis|0|1,custom|cmmlu:logical|0|1,custom|cmmlu:machine_learning|0|1,custom|cmmlu:management|0|1,custom|cmmlu:marketing|0|1,custom|cmmlu:marxist_theory|0|1,custom|cmmlu:modern_chinese|0|1,custom|cmmlu:nutrition|0|1,custom|cmmlu:philosophy|0|1,custom|cmmlu:professional_accounting|0|1,custom|cmmlu:professional_law|0|1,custom|cmmlu:professional_medicine|0|1,custom|cmmlu:professional_psychology|0|1,custom|cmmlu:public_relations|0|1,custom|cmmlu:security_study|0|1,custom|cmmlu:sociology|0|1,custom|cmmlu:sports_science|0|1,custom|cmmlu:traditional_chinese_medicine|0|1,custom|cmmlu:virology|0|1,custom|cmmlu:world_history|0|1,custom|cmmlu:world_religions|0|1
===================
custom|ceval:computer_network|0|1,custom|ceval:operating_system|0|1,custom|ceval:computer_architecture|0|1,custom|ceval:college_programming|0|1,custom|ceval:college_physics|0|1,custom|ceval:college_chemistry|0|1,custom|ceval:advanced_mathematics|0|1,custom|ceval:probability_and_statistics|0|1,custom|ceval:discrete_mathematics|0|1,custom|ceval:electrical_engineer|0|1,custom|ceval:metrology_engineer|0|1,custom|ceval:high_school_mathematics|0|1,custom|ceval:high_school_physics|0|1,custom|ceval:high_school_chemistry|0|1,custom|ceval:high_school_biology|0|1,custom|ceval:middle_school_mathematics|0|1,custom|ceval:middle_school_biology|0|1,custom|ceval:middle_school_physics|0|1,custom|ceval:middle_school_chemistry|0|1,custom|ceval:veterinary_medicine|0|1,custom|ceval:college_economics|0|1,custom|ceval:business_administration|0|1,custom|ceval:marxism|0|1,custom|ceval:mao_zedong_thought|0|1,custom|ceval:education_science|0|1,custom|ceval:teacher_qualification|0|1,custom|ceval:high_school_politics|0|1,custom|ceval:high_school_geography|0|1,custom|ceval:middle_school_politics|0|1,custom|ceval:middle_school_geography|0|1,custom|ceval:modern_chinese_history|0|1,custom|ceval:ideological_and_moral_cultivation|0|1,custom|ceval:logic|0|1,custom|ceval:law|0|1,custom|ceval:chinese_language_and_literature|0|1,custom|ceval:art_studies|0|1,custom|ceval:professional_tour_guide|0|1,custom|ceval:legal_professional|0|1,custom|ceval:high_school_chinese|0|1,custom|ceval:high_school_history|0|1,custom|ceval:middle_school_history|0|1,custom|ceval:civil_servant|0|1,custom|ceval:sports_science|0|1,custom|ceval:plant_protection|0|1,custom|ceval:basic_medicine|0|1,custom|ceval:clinical_medicine|0|1,custom|ceval:urban_and_rural_planner|0|1,custom|ceval:accountant|0|1,custom|ceval:fire_engineer|0|1,custom|ceval:environmental_impact_assessment_engineer|0|1,custom|ceval:tax_accountant|0|1,custom|ceval:physician|0|1
===================

More info here: https://github.com/huggingface/lighteval?tab=readme-ov-file#evaluate-a-model-on-extended-community-or-custom-tasks
For more info on differences between MMLU implementations: https://huggingface.co/blog/open-llm-leaderboard-mmlu#1001-flavors-of-mmlu
In particular, the default leaderboard MMLU implementation (which uses "A", "B", etc as answer targets) gives generally random results on small/non instruction tuned models.
Instead, we use the full MMLU answer as the target.
"""
import re
from typing import List, Tuple

from lighteval.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES

_TASKS_STRINGS: List[Tuple[LightevalTaskConfig, str]] = []
_TASKS: List[LightevalTaskConfig] = []

## COMMON_SENSE_REASONING_TASKS ##
COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function="hellaswag_prompt",
        hf_repo="hellaswag",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function="winogrande",
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function="piqa_harness",
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function="siqa_prompt",
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function="openbookqa",
        hf_repo="openbookqa",
        hf_subset="main",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function="commonsense_qa_prompt",
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
]


def commonsense_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def hellaswag_prompt(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )


# 0 short for common sense
COMMON_SENSE_REASONING_STRING = [(t, f"custom|{t.name}|0|1") for t in COMMON_SENSE_REASONING_TASKS]
_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)
_TASKS += COMMON_SENSE_REASONING_TASKS

## MMLU ##
class CustomMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="mmlu_prompt",
        hf_repo="lighteval/mmlu",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        suite=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


MMLU_TASKS = [
    CustomMMLUEvaluationTask(name="mmlu:abstract_algebra", hf_subset="abstract_algebra"),
    CustomMMLUEvaluationTask(name="mmlu:anatomy", hf_subset="anatomy"),
    CustomMMLUEvaluationTask(name="mmlu:astronomy", hf_subset="astronomy"),
    CustomMMLUEvaluationTask(name="mmlu:business_ethics", hf_subset="business_ethics"),
    CustomMMLUEvaluationTask(name="mmlu:clinical_knowledge", hf_subset="clinical_knowledge"),
    CustomMMLUEvaluationTask(name="mmlu:college_biology", hf_subset="college_biology"),
    CustomMMLUEvaluationTask(name="mmlu:college_chemistry", hf_subset="college_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:college_computer_science", hf_subset="college_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:college_mathematics", hf_subset="college_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:college_medicine", hf_subset="college_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:college_physics", hf_subset="college_physics"),
    CustomMMLUEvaluationTask(name="mmlu:computer_security", hf_subset="computer_security"),
    CustomMMLUEvaluationTask(name="mmlu:conceptual_physics", hf_subset="conceptual_physics"),
    CustomMMLUEvaluationTask(name="mmlu:econometrics", hf_subset="econometrics"),
    CustomMMLUEvaluationTask(name="mmlu:electrical_engineering", hf_subset="electrical_engineering"),
    CustomMMLUEvaluationTask(name="mmlu:elementary_mathematics", hf_subset="elementary_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:formal_logic", hf_subset="formal_logic"),
    CustomMMLUEvaluationTask(name="mmlu:global_facts", hf_subset="global_facts"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_biology", hf_subset="high_school_biology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_chemistry", hf_subset="high_school_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_computer_science", hf_subset="high_school_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_european_history", hf_subset="high_school_european_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_geography", hf_subset="high_school_geography"),
    CustomMMLUEvaluationTask(
        name="mmlu:high_school_government_and_politics", hf_subset="high_school_government_and_politics"
    ),
    CustomMMLUEvaluationTask(name="mmlu:high_school_macroeconomics", hf_subset="high_school_macroeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_mathematics", hf_subset="high_school_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_microeconomics", hf_subset="high_school_microeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_physics", hf_subset="high_school_physics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_psychology", hf_subset="high_school_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_statistics", hf_subset="high_school_statistics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_us_history", hf_subset="high_school_us_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_world_history", hf_subset="high_school_world_history"),
    CustomMMLUEvaluationTask(name="mmlu:human_aging", hf_subset="human_aging"),
    CustomMMLUEvaluationTask(name="mmlu:human_sexuality", hf_subset="human_sexuality"),
    CustomMMLUEvaluationTask(name="mmlu:international_law", hf_subset="international_law"),
    CustomMMLUEvaluationTask(name="mmlu:jurisprudence", hf_subset="jurisprudence"),
    CustomMMLUEvaluationTask(name="mmlu:logical_fallacies", hf_subset="logical_fallacies"),
    CustomMMLUEvaluationTask(name="mmlu:machine_learning", hf_subset="machine_learning"),
    CustomMMLUEvaluationTask(name="mmlu:management", hf_subset="management"),
    CustomMMLUEvaluationTask(name="mmlu:marketing", hf_subset="marketing"),
    CustomMMLUEvaluationTask(name="mmlu:medical_genetics", hf_subset="medical_genetics"),
    CustomMMLUEvaluationTask(name="mmlu:miscellaneous", hf_subset="miscellaneous"),
    CustomMMLUEvaluationTask(name="mmlu:moral_disputes", hf_subset="moral_disputes"),
    CustomMMLUEvaluationTask(name="mmlu:moral_scenarios", hf_subset="moral_scenarios"),
    CustomMMLUEvaluationTask(name="mmlu:nutrition", hf_subset="nutrition"),
    CustomMMLUEvaluationTask(name="mmlu:philosophy", hf_subset="philosophy"),
    CustomMMLUEvaluationTask(name="mmlu:prehistory", hf_subset="prehistory"),
    CustomMMLUEvaluationTask(name="mmlu:professional_accounting", hf_subset="professional_accounting"),
    CustomMMLUEvaluationTask(name="mmlu:professional_law", hf_subset="professional_law"),
    CustomMMLUEvaluationTask(name="mmlu:professional_medicine", hf_subset="professional_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:professional_psychology", hf_subset="professional_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:public_relations", hf_subset="public_relations"),
    CustomMMLUEvaluationTask(name="mmlu:security_studies", hf_subset="security_studies"),
    CustomMMLUEvaluationTask(name="mmlu:sociology", hf_subset="sociology"),
    CustomMMLUEvaluationTask(name="mmlu:us_foreign_policy", hf_subset="us_foreign_policy"),
    CustomMMLUEvaluationTask(name="mmlu:virology", hf_subset="virology"),
    CustomMMLUEvaluationTask(name="mmlu:world_religions", hf_subset="world_religions"),
]


def mmlu_prompt(line, task_name: str = None):
    """MMLU prompt without letters"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"
    #print(f"mmlu_prompt={prompt}")

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


MMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in MMLU_TASKS]
_TASKS_STRINGS.extend(MMLU_STRING)
_TASKS += MMLU_TASKS


############################################################################################################################################################
## CMMLU ##
class CustomCMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="cmmlu_prompt",
        hf_repo="ldwang/lighteval-cmmlu",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        suite=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
            trust_dataset=True,
        )


CMMLU_TASKS = [
		CustomCMMLUEvaluationTask(name="cmmlu:agronomy", hf_subset="agronomy"),
		CustomCMMLUEvaluationTask(name="cmmlu:anatomy", hf_subset="anatomy"),
		CustomCMMLUEvaluationTask(name="cmmlu:ancient_chinese", hf_subset="ancient_chinese"),
		CustomCMMLUEvaluationTask(name="cmmlu:arts", hf_subset="arts"),
		CustomCMMLUEvaluationTask(name="cmmlu:astronomy", hf_subset="astronomy"),
		CustomCMMLUEvaluationTask(name="cmmlu:business_ethics", hf_subset="business_ethics"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_civil_service_exam", hf_subset="chinese_civil_service_exam"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_driving_rule", hf_subset="chinese_driving_rule"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_food_culture", hf_subset="chinese_food_culture"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_foreign_policy", hf_subset="chinese_foreign_policy"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_history", hf_subset="chinese_history"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_literature", hf_subset="chinese_literature"),
		CustomCMMLUEvaluationTask(name="cmmlu:chinese_teacher_qualification", hf_subset="chinese_teacher_qualification"),
		CustomCMMLUEvaluationTask(name="cmmlu:clinical_knowledge", hf_subset="clinical_knowledge"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_actuarial_science", hf_subset="college_actuarial_science"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_education", hf_subset="college_education"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_engineering_hydrology", hf_subset="college_engineering_hydrology"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_law", hf_subset="college_law"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_mathematics", hf_subset="college_mathematics"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_medical_statistics", hf_subset="college_medical_statistics"),
		CustomCMMLUEvaluationTask(name="cmmlu:college_medicine", hf_subset="college_medicine"),
		CustomCMMLUEvaluationTask(name="cmmlu:computer_science", hf_subset="computer_science"),
		CustomCMMLUEvaluationTask(name="cmmlu:computer_security", hf_subset="computer_security"),
		CustomCMMLUEvaluationTask(name="cmmlu:conceptual_physics", hf_subset="conceptual_physics"),
		CustomCMMLUEvaluationTask(name="cmmlu:construction_project_management", hf_subset="construction_project_management"),
		CustomCMMLUEvaluationTask(name="cmmlu:economics", hf_subset="economics"),
		CustomCMMLUEvaluationTask(name="cmmlu:education", hf_subset="education"),
		CustomCMMLUEvaluationTask(name="cmmlu:electrical_engineering", hf_subset="electrical_engineering"),
		CustomCMMLUEvaluationTask(name="cmmlu:elementary_chinese", hf_subset="elementary_chinese"),
		CustomCMMLUEvaluationTask(name="cmmlu:elementary_commonsense", hf_subset="elementary_commonsense"),
		CustomCMMLUEvaluationTask(name="cmmlu:elementary_information_and_technology", hf_subset="elementary_information_and_technology"),
		CustomCMMLUEvaluationTask(name="cmmlu:elementary_mathematics", hf_subset="elementary_mathematics"),
		CustomCMMLUEvaluationTask(name="cmmlu:ethnology", hf_subset="ethnology"),
		CustomCMMLUEvaluationTask(name="cmmlu:food_science", hf_subset="food_science"),
		CustomCMMLUEvaluationTask(name="cmmlu:genetics", hf_subset="genetics"),
		CustomCMMLUEvaluationTask(name="cmmlu:global_facts", hf_subset="global_facts"),
		CustomCMMLUEvaluationTask(name="cmmlu:high_school_biology", hf_subset="high_school_biology"),
		CustomCMMLUEvaluationTask(name="cmmlu:high_school_chemistry", hf_subset="high_school_chemistry"),
		CustomCMMLUEvaluationTask(name="cmmlu:high_school_geography", hf_subset="high_school_geography"),
		CustomCMMLUEvaluationTask(name="cmmlu:high_school_mathematics", hf_subset="high_school_mathematics"),
		CustomCMMLUEvaluationTask(name="cmmlu:high_school_physics", hf_subset="high_school_physics"),
		CustomCMMLUEvaluationTask(name="cmmlu:high_school_politics", hf_subset="high_school_politics"),
		CustomCMMLUEvaluationTask(name="cmmlu:human_sexuality", hf_subset="human_sexuality"),
		CustomCMMLUEvaluationTask(name="cmmlu:international_law", hf_subset="international_law"),
		CustomCMMLUEvaluationTask(name="cmmlu:journalism", hf_subset="journalism"),
		CustomCMMLUEvaluationTask(name="cmmlu:jurisprudence", hf_subset="jurisprudence"),
		CustomCMMLUEvaluationTask(name="cmmlu:legal_and_moral_basis", hf_subset="legal_and_moral_basis"),
		CustomCMMLUEvaluationTask(name="cmmlu:logical", hf_subset="logical"),
		CustomCMMLUEvaluationTask(name="cmmlu:machine_learning", hf_subset="machine_learning"),
		CustomCMMLUEvaluationTask(name="cmmlu:management", hf_subset="management"),
		CustomCMMLUEvaluationTask(name="cmmlu:marketing", hf_subset="marketing"),
		CustomCMMLUEvaluationTask(name="cmmlu:marxist_theory", hf_subset="marxist_theory"),
		CustomCMMLUEvaluationTask(name="cmmlu:modern_chinese", hf_subset="modern_chinese"),
		CustomCMMLUEvaluationTask(name="cmmlu:nutrition", hf_subset="nutrition"),
		CustomCMMLUEvaluationTask(name="cmmlu:philosophy", hf_subset="philosophy"),
		CustomCMMLUEvaluationTask(name="cmmlu:professional_accounting", hf_subset="professional_accounting"),
		CustomCMMLUEvaluationTask(name="cmmlu:professional_law", hf_subset="professional_law"),
		CustomCMMLUEvaluationTask(name="cmmlu:professional_medicine", hf_subset="professional_medicine"),
		CustomCMMLUEvaluationTask(name="cmmlu:professional_psychology", hf_subset="professional_psychology"),
		CustomCMMLUEvaluationTask(name="cmmlu:public_relations", hf_subset="public_relations"),
		CustomCMMLUEvaluationTask(name="cmmlu:security_study", hf_subset="security_study"),
		CustomCMMLUEvaluationTask(name="cmmlu:sociology", hf_subset="sociology"),
		CustomCMMLUEvaluationTask(name="cmmlu:sports_science", hf_subset="sports_science"),
		CustomCMMLUEvaluationTask(name="cmmlu:traditional_chinese_medicine", hf_subset="traditional_chinese_medicine"),
		CustomCMMLUEvaluationTask(name="cmmlu:virology", hf_subset="virology"),
		CustomCMMLUEvaluationTask(name="cmmlu:world_history", hf_subset="world_history"),
		CustomCMMLUEvaluationTask(name="cmmlu:world_religions", hf_subset="world_religions"),
]

cmmlu_subject_mapping = {
    'agronomy': '农学',
    'anatomy': '解剖学',
    'ancient_chinese': '古汉语',
    'arts': '艺术学',
    'astronomy': '天文学',
    'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '中国公务员考试',
    'chinese_driving_rule': '中国驾驶规则',
    'chinese_food_culture': '中国饮食文化',
    'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史',
    'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '中国教师资格',
    'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学',
    'college_education': '大学教育学',
    'college_engineering_hydrology': '大学工程水文学',
    'college_law': '大学法律',
    'college_mathematics': '大学数学',
    'college_medical_statistics': '大学医学统计',
    'college_medicine': '大学医学',
    'computer_science': '计算机科学',
    'computer_security': '计算机安全',
    'conceptual_physics': '概念物理学',
    'construction_project_management': '建设工程管理',
    'economics': '经济学',
    'education': '教育学',
    'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文',
    'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术',
    'elementary_mathematics': '初等数学',
    'ethnology': '民族学',
    'food_science': '食品科学',
    'genetics': '遗传学',
    'global_facts': '全球事实',
    'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学',
    'high_school_geography': '高中地理',
    'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理学',
    'high_school_politics': '高中政治',
    'human_sexuality': '人类性行为',
    'international_law': '国际法学',
    'journalism': '新闻学',
    'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础',
    'logical': '逻辑学',
    'machine_learning': '机器学习',
    'management': '管理学',
    'marketing': '市场营销',
    'marxist_theory': '马克思主义理论',
    'modern_chinese': '现代汉语',
    'nutrition': '营养学',
    'philosophy': '哲学',
    'professional_accounting': '专业会计',
    'professional_law': '专业法学',
    'professional_medicine': '专业医学',
    'professional_psychology': '专业心理学',
    'public_relations': '公共关系',
    'security_study': '安全研究',
    'sociology': '社会学',
    'sports_science': '体育学',
    'traditional_chinese_medicine': '中医中药',
    'virology': '病毒学',
    'world_history': '世界历史',
    'world_religions': '世界宗教'
}

def cmmlu_prompt(line, task_name: str = None):
    # 以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}
    # 答案是: {{{answer}}}
    """CMMLU prompt without letters"""
    topic = cmmlu_subject_mapping[line['subject']]
    prompt = f"以下是关于{topic.replace('_', ' ')}的单项选择题，请直接给出正确答案的选项。\n题目："
    prompt += line["question"] + "\n答案是："
    #print(f"cmmlu_prompt={prompt}")

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=None,
    )

CMMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in CMMLU_TASKS]
_TASKS_STRINGS.extend(CMMLU_STRING)
_TASKS += CMMLU_TASKS
print(f'{",".join([t[1] for t in CMMLU_STRING])}')

############################################################################################################################################################
## CEVAL ##
class CustomCEVALEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="ceval_prompt",
        hf_repo="ldwang/lighteval-ceval-exam",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["val"],
        few_shots_split="dev",
        few_shots_select=None,
        suite=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
            trust_dataset=True,
        )


CEVAL_TASKS = [
    CustomCEVALEvaluationTask(name="ceval:computer_network", hf_subset="computer_network"),
    CustomCEVALEvaluationTask(name="ceval:operating_system", hf_subset="operating_system"),
    CustomCEVALEvaluationTask(name="ceval:computer_architecture", hf_subset="computer_architecture"),
    CustomCEVALEvaluationTask(name="ceval:college_programming", hf_subset="college_programming"),
    CustomCEVALEvaluationTask(name="ceval:college_physics", hf_subset="college_physics"),
    CustomCEVALEvaluationTask(name="ceval:college_chemistry", hf_subset="college_chemistry"),
    CustomCEVALEvaluationTask(name="ceval:advanced_mathematics", hf_subset="advanced_mathematics"),
    CustomCEVALEvaluationTask(name="ceval:probability_and_statistics", hf_subset="probability_and_statistics"),
    CustomCEVALEvaluationTask(name="ceval:discrete_mathematics", hf_subset="discrete_mathematics"),
    CustomCEVALEvaluationTask(name="ceval:electrical_engineer", hf_subset="electrical_engineer"),
    CustomCEVALEvaluationTask(name="ceval:metrology_engineer", hf_subset="metrology_engineer"),
    CustomCEVALEvaluationTask(name="ceval:high_school_mathematics", hf_subset="high_school_mathematics"),
    CustomCEVALEvaluationTask(name="ceval:high_school_physics", hf_subset="high_school_physics"),
    CustomCEVALEvaluationTask(name="ceval:high_school_chemistry", hf_subset="high_school_chemistry"),
    CustomCEVALEvaluationTask(name="ceval:high_school_biology", hf_subset="high_school_biology"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_mathematics", hf_subset="middle_school_mathematics"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_biology", hf_subset="middle_school_biology"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_physics", hf_subset="middle_school_physics"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_chemistry", hf_subset="middle_school_chemistry"),
    CustomCEVALEvaluationTask(name="ceval:veterinary_medicine", hf_subset="veterinary_medicine"),
    CustomCEVALEvaluationTask(name="ceval:college_economics", hf_subset="college_economics"),
    CustomCEVALEvaluationTask(name="ceval:business_administration", hf_subset="business_administration"),
    CustomCEVALEvaluationTask(name="ceval:marxism", hf_subset="marxism"),
    CustomCEVALEvaluationTask(name="ceval:mao_zedong_thought", hf_subset="mao_zedong_thought"),
    CustomCEVALEvaluationTask(name="ceval:education_science", hf_subset="education_science"),
    CustomCEVALEvaluationTask(name="ceval:teacher_qualification", hf_subset="teacher_qualification"),
    CustomCEVALEvaluationTask(name="ceval:high_school_politics", hf_subset="high_school_politics"),
    CustomCEVALEvaluationTask(name="ceval:high_school_geography", hf_subset="high_school_geography"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_politics", hf_subset="middle_school_politics"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_geography", hf_subset="middle_school_geography"),
    CustomCEVALEvaluationTask(name="ceval:modern_chinese_history", hf_subset="modern_chinese_history"),
    CustomCEVALEvaluationTask(name="ceval:ideological_and_moral_cultivation", hf_subset="ideological_and_moral_cultivation"),
    CustomCEVALEvaluationTask(name="ceval:logic", hf_subset="logic"),
    CustomCEVALEvaluationTask(name="ceval:law", hf_subset="law"),
    CustomCEVALEvaluationTask(name="ceval:chinese_language_and_literature", hf_subset="chinese_language_and_literature"),
    CustomCEVALEvaluationTask(name="ceval:art_studies", hf_subset="art_studies"),
    CustomCEVALEvaluationTask(name="ceval:professional_tour_guide", hf_subset="professional_tour_guide"),
    CustomCEVALEvaluationTask(name="ceval:legal_professional", hf_subset="legal_professional"),
    CustomCEVALEvaluationTask(name="ceval:high_school_chinese", hf_subset="high_school_chinese"),
    CustomCEVALEvaluationTask(name="ceval:high_school_history", hf_subset="high_school_history"),
    CustomCEVALEvaluationTask(name="ceval:middle_school_history", hf_subset="middle_school_history"),
    CustomCEVALEvaluationTask(name="ceval:civil_servant", hf_subset="civil_servant"),
    CustomCEVALEvaluationTask(name="ceval:sports_science", hf_subset="sports_science"),
    CustomCEVALEvaluationTask(name="ceval:plant_protection", hf_subset="plant_protection"),
    CustomCEVALEvaluationTask(name="ceval:basic_medicine", hf_subset="basic_medicine"),
    CustomCEVALEvaluationTask(name="ceval:clinical_medicine", hf_subset="clinical_medicine"),
    CustomCEVALEvaluationTask(name="ceval:urban_and_rural_planner", hf_subset="urban_and_rural_planner"),
    CustomCEVALEvaluationTask(name="ceval:accountant", hf_subset="accountant"),
    CustomCEVALEvaluationTask(name="ceval:fire_engineer", hf_subset="fire_engineer"),
    CustomCEVALEvaluationTask(name="ceval:environmental_impact_assessment_engineer", hf_subset="environmental_impact_assessment_engineer"),
    CustomCEVALEvaluationTask(name="ceval:tax_accountant", hf_subset="tax_accountant"),
    CustomCEVALEvaluationTask(name="ceval:physician", hf_subset="physician"),
]

ceval_subject_mapping = {
    'computer_network': ['Computer Network', '计算机网络', 'STEM'],
    'operating_system': ['Operating System', '操作系统', 'STEM'],
    'computer_architecture': ['Computer Architecture', '计算机组成', 'STEM'],
    'college_programming': ['College Programming', '大学编程', 'STEM'],
    'college_physics': ['College Physics', '大学物理', 'STEM'],
    'college_chemistry': ['College Chemistry', '大学化学', 'STEM'],
    'advanced_mathematics': ['Advanced Mathematics', '高等数学', 'STEM'],
    'probability_and_statistics': ['Probability and Statistics', '概率统计', 'STEM'],
    'discrete_mathematics': ['Discrete Mathematics', '离散数学', 'STEM'],
    'electrical_engineer': ['Electrical Engineer', '注册电气工程师', 'STEM'],
    'metrology_engineer': ['Metrology Engineer', '注册计量师', 'STEM'],
    'high_school_mathematics': ['High School Mathematics', '高中数学', 'STEM'],
    'high_school_physics': ['High School Physics', '高中物理', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', '高中化学', 'STEM'],
    'high_school_biology': ['High School Biology', '高中生物', 'STEM'],
    'middle_school_mathematics': ['Middle School Mathematics', '初中数学', 'STEM'],
    'middle_school_biology': ['Middle School Biology', '初中生物', 'STEM'],
    'middle_school_physics': ['Middle School Physics', '初中物理', 'STEM'],
    'middle_school_chemistry': ['Middle School Chemistry', '初中化学', 'STEM'],
    'veterinary_medicine': ['Veterinary Medicine', '兽医学', 'STEM'],
    'college_economics': ['College Economics', '大学经济学', 'Social Science'],
    'business_administration': ['Business Administration', '工商管理', 'Social Science'],
    'marxism': ['Marxism', '马克思主义基本原理', 'Social Science'],
    'mao_zedong_thought': ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science'],
    'education_science': ['Education Science', '教育学', 'Social Science'],
    'teacher_qualification': ['Teacher Qualification', '教师资格', 'Social Science'],
    'high_school_politics': ['High School Politics', '高中政治', 'Social Science'],
    'high_school_geography': ['High School Geography', '高中地理', 'Social Science'],
    'middle_school_politics': ['Middle School Politics', '初中政治', 'Social Science'],
    'middle_school_geography': ['Middle School Geography', '初中地理', 'Social Science'],
    'modern_chinese_history': ['Modern Chinese History', '近代史纲要', 'Humanities'],
    'ideological_and_moral_cultivation': ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities'],
    'logic': ['Logic', '逻辑学', 'Humanities'],
    'law': ['Law', '法学', 'Humanities'],
    'chinese_language_and_literature': ['Chinese Language and Literature', '中国语言文学', 'Humanities'],
    'art_studies': ['Art Studies', '艺术学', 'Humanities'],
    'professional_tour_guide': ['Professional Tour Guide', '导游资格', 'Humanities'],
    'legal_professional': ['Legal Professional', '法律职业资格', 'Humanities'],
    'high_school_chinese': ['High School Chinese', '高中语文', 'Humanities'],
    'high_school_history': ['High School History', '高中历史', 'Humanities'],
    'middle_school_history': ['Middle School History', '初中历史', 'Humanities'],
    'civil_servant': ['Civil Servant', '公务员', 'Other'],
    'sports_science': ['Sports Science', '体育学', 'Other'],
    'plant_protection': ['Plant Protection', '植物保护', 'Other'],
    'basic_medicine': ['Basic Medicine', '基础医学', 'Other'],
    'clinical_medicine': ['Clinical Medicine', '临床医学', 'Other'],
    'urban_and_rural_planner': ['Urban and Rural Planner', '注册城乡规划师', 'Other'],
    'accountant': ['Accountant', '注册会计师', 'Other'],
    'fire_engineer': ['Fire Engineer', '注册消防工程师', 'Other'],
    'environmental_impact_assessment_engineer': ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other'],
    'tax_accountant': ['Tax Accountant', '税务师', 'Other'],
    'physician': ['Physician', '医师资格', 'Other'],
}

def ceval_prompt(line, task_name: str = None):
    # f"以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: "
    """CEVAL prompt without letters"""
    topic = ceval_subject_mapping[line['subject']][1]
    prompt = f"以下是中国关于{topic.replace('_', ' ')}考试的单项选择题，请选出其中的正确答案。\n题目："
    prompt += line["question"] + "\n答案："
    #print(f"ceval_prompt={prompt}")

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=None,
    )

CEVAL_STRING = [(t, f"custom|{t.name}|0|1") for t in CEVAL_TASKS]
_TASKS_STRINGS.extend(CEVAL_STRING)
_TASKS += CEVAL_TASKS
print(f'{",".join([t[1] for t in CEVAL_STRING])}')

############################################################################################################################################################

# common sense reasoning + mmlu
EARLY_SIGNAL_TASKS = ",".join([t[1] for t in COMMON_SENSE_REASONING_STRING] + [t[1] for t in MMLU_STRING] + [t[1] for t in CMMLU_STRING])

# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]
# You can have a few pre-organised groups of tasks
TASKS_GROUPS = {
    "early-signal": EARLY_SIGNAL_TASKS,
}
