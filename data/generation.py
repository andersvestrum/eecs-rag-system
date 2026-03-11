"""Generate synthetic validation Q&A files for local testing. Run from repo root: python data/generation.py"""

import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

courses = ["CS 61A", "CS 61B", "CS 61C", "CS 70", "CS 188", "CS 189"]
labs = ["BAIR Lab", "RISELab", "Berkeley DeepDrive"]
buildings = ["Soda Hall", "Cory Hall", "Jacobs Hall"]
faculty = ["Pieter Abbeel", "Dawn Song", "Ion Stoica"]
orgs = ["Machine Learning at Berkeley", "IEEE Berkeley"]

q_templates = [
    "What topics are covered in {course}?",
    "What research does the {lab} focus on?",
    "Where is {building} located on the UC Berkeley campus?",
    "What does Professor {prof} research at Berkeley?",
    "How can students join {org}?",
]

a_templates = [
    "{course} is an EECS course at UC Berkeley covering core computer science concepts.",
    "The {lab} conducts research in artificial intelligence and distributed systems.",
    "{building} is one of the main buildings used by the EECS department.",
    "Professor {prof} is an EECS faculty member conducting advanced research.",
    "{org} is a Berkeley student organization focused on engineering and computer science.",
]

questions = []
answers = []

for _ in range(1000):
    course = random.choice(courses)
    lab = random.choice(labs)
    building = random.choice(buildings)
    prof = random.choice(faculty)
    org = random.choice(orgs)

    # Use the same template index for Q and A so each pair matches
    i = random.randrange(len(q_templates))
    q = q_templates[i].format(
        course=course, lab=lab, building=building, prof=prof, org=org
    )
    a = a_templates[i].format(
        course=course, lab=lab, building=building, prof=prof, org=org
    )

    questions.append(q)
    answers.append(a)

out_questions = os.path.join(SCRIPT_DIR, "validation_questions.txt")
out_answers = os.path.join(SCRIPT_DIR, "validation_answers.txt")

with open(out_questions, "w") as f:
    f.write("\n".join(questions) + "\n")
with open(out_answers, "w") as f:
    f.write("\n".join(answers) + "\n")

print(f"Wrote {len(questions)} questions -> {out_questions}")
print(f"Wrote {len(answers)} answers   -> {out_answers}")