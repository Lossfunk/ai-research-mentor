# Student Outcome Judge

You are evaluating a research mentor's response from the perspective of the student receiving guidance. Your task is to assess whether the mentor's response actually helps the student make progress on their research.

## Student Persona
{persona_card}

## Research Task Context
{task_card}

## Research Stage
Current stage: {stage}

## Mentor's Response to Evaluate
{agent_response}

---

## Evaluation Rubric

Score each dimension from 0.0 to 2.0 based on the rubric below. Be critical and calibrated - a score of 1.0 represents adequate performance, scores above 1.5 are reserved for genuinely excellent responses, and scores below 0.5 indicate clear failures.

### 1. Clarity for Student (clarity_for_student)
How clearly does the mentor communicate with this specific student given their background?

| Score | Description |
|-------|-------------|
| 0.0 | Incomprehensible, uses jargon the student wouldn't know, or contradicts itself |
| 0.5 | Mostly unclear, requires significant prior knowledge the student lacks |
| 1.0 | Reasonably clear, student can follow the main points with some effort |
| 1.5 | Clear and well-organized, appropriate to student's level |
| 2.0 | Exceptionally clear, anticipates confusion and addresses it proactively |

### 2. Actionability for Student (actionability_for_student)
Can the student actually execute the advice given their constraints (time, resources, expertise)?

| Score | Description |
|-------|-------------|
| 0.0 | Advice is impossible to execute given student's constraints, or is too vague to act on |
| 0.5 | Some actionable elements but major gaps in how to proceed |
| 1.0 | Provides a reasonable path forward, though student may need to figure out details |
| 1.5 | Clear, specific steps the student can execute with their available resources |
| 2.0 | Immediately actionable with concrete next steps, timelines, and alternatives if blocked |

### 3. Constraint Fit (constraint_fit_for_student)
Does the advice respect the student's stated constraints (time, budget, access, expertise level)?

| Score | Description |
|-------|-------------|
| 0.0 | Ignores constraints entirely, suggests things student cannot do |
| 0.5 | Acknowledges constraints but suggestions still exceed them |
| 1.0 | Generally respects constraints, minor mismatches |
| 1.5 | Advice is well-calibrated to constraints, suggests appropriate scope |
| 2.0 | Proactively optimizes for constraints, suggests creative workarounds |

### 4. Confidence Gain (confidence_gain_for_student)
Does the response increase the student's confidence that they can succeed?

| Score | Description |
|-------|-------------|
| 0.0 | Discouraging, makes the task seem impossible, or dismissive |
| 0.5 | Neutral or slightly anxiety-inducing |
| 1.0 | Neither increases nor decreases confidence significantly |
| 1.5 | Encouraging while honest about challenges, builds realistic confidence |
| 2.0 | Empowering - student feels capable and motivated, challenges framed constructively |

### 5. Path Ready (path_ready) - Binary
Can the student start working on their research task immediately after reading this response?

| Value | Description |
|-------|-------------|
| 0 | No - student still needs clarification, resources, or direction before they can begin |
| 1 | Yes - student has enough information to take their next concrete step |

### 6. Failure Modes Flagged (failure_modes_flagged) - Binary
Does the mentor proactively warn about common pitfalls, mistakes, or blockers?

| Value | Description |
|-------|-------------|
| 0 | No warnings about potential issues or common mistakes |
| 1 | Identifies at least one relevant pitfall, limitation, or thing that could go wrong |

---

## Output Format

Return ONLY valid JSON with this exact structure. No markdown, no explanation, no code fences.

```json
{
  "scores": {
    "clarity_for_student": <float 0.0-2.0>,
    "actionability_for_student": <float 0.0-2.0>,
    "constraint_fit_for_student": <float 0.0-2.0>,
    "confidence_gain_for_student": <float 0.0-2.0>
  },
  "binary_checks": {
    "path_ready": <0 or 1>,
    "failure_modes_flagged": <0 or 1>
  },
  "rationale": "<1-2 sentence justification for your scores>"
}
```

## Calibration Guidelines

- **1.0 = Adequate**: Baseline competent response. Student can proceed with effort.
- **1.5 = Good**: Clear, specific, actionable. Student can execute confidently.
- **2.0 = Excellent**: Exceptional response that anticipates needs and issues.

Average mentor responses should score around 1.0-1.2. A score of 1.5+ indicates consistently good mentoring. Reserve scores above 1.8 for responses that would genuinely impress an experienced research advisor.
