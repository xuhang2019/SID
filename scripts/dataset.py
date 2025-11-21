from datasets import load_dataset
import random
from typing import Dict, Any, Tuple, Optional



def get_dataset_class(dataset_name, max_questions, start_idx, seed, subset):
    if dataset_name == "mmlupro":
        return MMLUProDataset(max_questions=max_questions, start_idx=start_idx, seed=seed, subset=subset)
    elif dataset_name == "scienceqa":
        return ScienceQADataset(max_questions=max_questions, start_idx=start_idx, seed=seed, subset=subset)
    elif dataset_name == "gpqa":
        return GPQADataset(subset=subset, max_questions=max_questions, start_idx=start_idx, seed=seed)
    elif dataset_name == "math":
        return MathDataset(subset=subset, max_questions=max_questions, start_idx=start_idx, seed=seed)
    elif dataset_name == "mmstar":
        return MMStarDataset(max_questions=max_questions, start_idx=start_idx, seed=seed, subset=subset)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

class BaseDataset:
    """Simple base dataset class defining the interface."""

    def __init__(
        self,
        split: str = "test",
        max_questions: Optional[int] = None,
        start_idx: Optional[int] = 0,
        seed: Optional[int] = 0,
        subset: Optional[str] = None,
    ) -> None:
        self.split = split
        self.max_questions = max_questions
        self.start_idx = start_idx
        self.seed = seed
        self.subset = subset

    def load(self):
        raise NotImplementedError

    def parse(self, example: Dict[str, Any]):
        """
            Parse the example into a tuple contains at least 2 elements, questions and answers.
        """
        raise NotImplementedError

    @staticmethod
    def _apply_sampling(dataset, max_questions=100, start_idx=0, seed=None):
        """
            max_questions: if None, return the whole dataset.
            start_idx: if >0, start from this index (useful for resuming).
            seed: if not None, sample randomly; otherwise, select sequentially.
        """
        total_len = len(dataset)
        if max_questions is None or max_questions >= total_len:
            max_questions = total_len

        # If seed is not None, do random sampling
        if seed is not None:
            random.seed(seed)
            indices = random.sample(range(total_len), max_questions)
            indices = indices[start_idx : start_idx + max_questions]
            return dataset.select(indices)
        else:
            return dataset.select(range(start_idx, min(max_questions, total_len)))


class MMLUProDataset(BaseDataset):
    def load(self):
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=self.split)
        return self._apply_sampling(dataset, self.max_questions, self.start_idx, self.seed)

    def parse(self, example: Dict[str, Any]) -> Tuple[str, str]:
        question = example['question']
        options = example['options']
        answer = example['answer']

        options_str = ""
        for i, option in enumerate(options):
            option_letter = chr(65 + i) if i < 10 else str(i)
            options_str += f"({option_letter}) {option}, "
        options_str = options_str.rstrip(", ")

        question_fmt = f"Can you answer the following question? {question}: {options_str}."
        return question_fmt, answer


class ScienceQADataset(BaseDataset):
    def load(self):
        #TODO: hard code for now
        dataset = load_dataset("derek-thomas/ScienceQA", split=self.split)
        filtered = []
        count = 0
        for example in dataset:
            if example['image'] is not None:
                filtered.append(example)
                count += 1
                if self.max_questions is not None and count >= self.max_questions:
                    break
        return filtered[self.start_idx:]

    def parse(self, example: Dict[str, Any]):
        use_lecture = True
        question = example['question']
        image = example['image']
        choices = example['choices']
        
        lecture, hint = None, None
        if use_lecture:
            lecture = example['lecture']
            hint = example['hint']
        
        question = create_prompt(question, choices, lecture, hint)
        
        # answer change to A, B
        answer = chr(65 + int(example['answer']))
        
        subject = example['subject']
        category = example.get('category', 'unknown')
        return image, question, answer, subject
    
class MMStarDataset(BaseDataset):
    """
        Brute-force set seed to 0
    """
    def load(self):
        #TODO: hard code for now
        dataset = load_dataset("Lin-Chen/MMStar", split='val')
        return self._apply_sampling(dataset, self.max_questions, self.start_idx, 0)

    def parse(self, example: Dict[str, Any]):
        question = example['question']
        image = example['image']
        answer = example['answer']
        category = example.get('category', 'unknown')
        return image, question, answer, category


class GPQADataset(BaseDataset):
    def __init__(
        self,
        subset: str = "gpqa_main",
        split: str = "train",
        max_questions: Optional[int] = 100,
        start_idx: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(split=split, max_questions=max_questions, start_idx=start_idx, seed=seed, subset=subset)

    def load(self):
        subset_name = self.subset or "gpqa_main"
        dataset = load_dataset("Idavidrein/gpqa", subset_name, split=self.split)
        return self._apply_sampling(dataset, self.max_questions, self.start_idx, self.seed)

    def parse(self, example: Dict[str, Any]):
        if self.seed is not None:
            random.seed(self.seed)

        question = example['Question']
        correct = example['Correct Answer']
        incorrects = [
            example['Incorrect Answer 1'],
            example['Incorrect Answer 2'],
            example['Incorrect Answer 3'],
        ]
        options = [(correct, True)] + [(opt, False) for opt in incorrects]
        random.shuffle(options)

        letters = ['A', 'B', 'C', 'D']
        opts_str = ""
        answer_letter = 'A'
        for i, (opt_text, is_correct) in enumerate(options):
            letter = letters[i]
            opts_str += f"({letter}) {opt_text} "
            if is_correct:
                answer_letter = letter

        prompt = (
            f"Can you answer the following question? {question} "
            f"Choices: {opts_str.strip()}. \n"
        )
        category = example['High-level domain']
        return prompt, answer_letter, category


class MathDataset(BaseDataset):
    def __init__(
        self,
        subset: str = "algebra",
        split: str = "test",
        max_questions: Optional[int] = None,
        start_idx: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
            subset: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
        """
        super().__init__(split=split, max_questions=max_questions, start_idx=start_idx, seed=seed, subset=subset)

    def load(self):
        dataset = load_dataset("EleutherAI/hendrycks_math", self.subset, split=self.split)
        return self._apply_sampling(dataset, self.max_questions, self.start_idx, self.seed)

    def parse(self, example: Dict[str, Any]):
        question = example['problem']
        answer = example['solution']
        category = example['type']
        level = example['level']
        prompt = "Given a mathematics problem, determine the answer. \nQuestion:" + question + "\n"
        return prompt, answer, category, level


def select_dataset(name: str, **kwargs) -> BaseDataset:
    """Factory function to create a dataset class instance.

    Args:
        name: One of 'mmlupro', 'scienceqa', 'gpqa', 'math'.
        kwargs: Parameters similar to the previous get_xxx_dataset functions.
    """
    key = (name or '').lower()
    if key in ["mmlu", "mmlupro", "mmlu-pro", "mmlu_pro"]:
        return MMLUProDataset(**kwargs)
    if key in ["scienceqa", "sqa"]:
        return ScienceQADataset(**kwargs)
    if key in ["gpqa"]:
        return GPQADataset(**kwargs)
    if key in ["math", "hendrycks_math", "hendrycks-math"]:
        return MathDataset(**kwargs)
    raise ValueError(f"Unknown dataset name: {name}")


# Compatibility wrapper functions to avoid changing existing callers
def get_mmlupro_dataset(split: str = "test", max_questions: Optional[int] = None, start_idx=None, seed: int = 0):
    ds = MMLUProDataset(split=split, max_questions=max_questions, start_idx=start_idx, seed=seed)
    return ds.load()


def parse_question_answer_mmlupro(example: Dict[str, Any]) -> Tuple[str, str]:
    return MMLUProDataset().parse(example)


def get_scienceqa_dataset(split: str = "test", max_questions: int = 100, start_idx: int = 0, seed=None):
    ds = ScienceQADataset(split=split, max_questions=max_questions, start_idx=start_idx, seed=seed)
    return ds.load()


def parse_question_answer_scienceqa(example):
    return ScienceQADataset().parse(example)


def get_gpqa_dataset(subset: str = "gpqa_main", split: str = "train", max_questions: int = 100, start_idx: int = 0, seed=None):
    ds = GPQADataset(subset=subset, split=split, max_questions=max_questions, start_idx=start_idx, seed=seed)
    return ds.load()


def parse_question_answer_gpqa(example, seed=0):
    return GPQADataset(seed=seed).parse(example)


def get_math_dataset(subset: str = "algebra", split: str = "test", max_questions: Optional[int] = None, start_idx: int = 0, seed=None):
    ds = MathDataset(subset=subset, split=split, max_questions=max_questions, start_idx=start_idx, seed=seed)
    return ds.load()


def parse_question_answer_math(example):
    return MathDataset().parse(example)


def create_options(options):
    if options != None:
        letters = ["(A) ", "(B) ", "(C) ", "(D) ", "(E) ", "(F) ", "(G) "]
        strs = "Options:\n"
        for i in range(len(options)):
            strs += letters[i]
            strs += options[i]
            strs += "\n"
        strs += "\n"
    else:
        strs = ""
    return strs


def create_lecture(lecture = None):
    strs = ""
    if lecture != None:
        strs = "Lecture:\n" + lecture + "\n\n"
    return strs


def create_context(context = None):
    strs = ""
    if context != None and context != "":
        strs = "Context:\n" + context + "\n\n"
    return strs


def create_prompt(question, options = None, context = None, lecture = None, if_options = True, output_format = None):
    prompt = "Question:\n" + question + "\n\n"
    if if_options and options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)
        if output_format is not None:
            prompt += output_format
    
    return prompt
    