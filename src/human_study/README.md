# Guide to Labeling Code Changes

## What You'll Be Doing

You'll be looking at **code changes** (shown as diffs) and answering two simple questions about each change:
1. **Does the changed program behave identically to the original program for all inputs?**
2. **On a scale of 1-5, how realistic is this code change?**

## The Two Questions You'll Answer

### Question 1: Does the changed program behave identically to the original for all inputs?

**Look for changes that would make the program do something different:** If the change is in code that never executes, or if it's mathematically equivalent, it doesn't affect behavior.

**✅ The programs have identical behavior (Answer 1):**
```python
# Original
if False:
    print("This never runs")

# Changed version
if False:
    print("This STILL never runs")  # Same behavior - dead code
```

**❌ Change to program behavior (Answer 0):**
```python
# Original  
if x < 10:
    return True

# Changed version  
if x <= 10:  # Now includes x=10, different behavior
    return True
```

```python
# Original
import pandas as pd

# Changed version - Commenting out an import causes change in behavior if the import is used
# import pandas as pd   
```

### Question 2: On a scale of 1-5, does this look like a realistic developer mistake?

**Think: "Could someone accidentally write this while coding?"**

**✅ Realistic mistake (Answer 4 or 5):**
```python
# Original
for i in range(len(items)):

# Changed version - Natural off-by-one error
for i in range(len(items) - 1):
```

**❌ Unrealistic change (Answer 1 or 2):**
```python
# Original
def calculate_total(prices):

# Changed version - No developer would write this
def xkcd_random_gibberish_name(prices):
```

```python
# Original
result = a + b

# Changed version - meaningless Python syntax
result = a @@ b
```

**Key insight:** Ask yourself "Would a real person accidentally type this?" Typos and common mistakes = realistic. Random gibberish = unrealistic.

## Step-by-Step Labeling Process

### Step 1: Setup
1. Install the repository and dependencies
```bash
git clone https://github.com/Jirachiii/mutant_analysis.git
pip install requests colorama
```
2. Save your input file (e.g., `test_sampled_mutants.json`) inside the repository.
3. Update the `filename` variable at line 7 of `object_browser.py` to match your input file name.
4. Run the labeling program
```bash
cd mutant_analysis
python object_browser.py
```

### Step 2: Analyze Each Sample
1. **Examine the diff** - A comparison between original and mutant code is displayed, such as:
```
    def __call__(self, data, groupby, orient, scales):

-        return (
-            groupby
-            .apply(data.dropna(subset=["x", "y"]), self._fit_predict)
-        )
+        return (
+            groupby
+            .apply(data, self._fit_predict)
+        )
```

2. **Understand the change** - In this example: The change takes place in line ```.apply(data.dropna(subset=["x", "y"]), self._fit_predict)```, which was changed to ```.apply(data, self._fit_predict)```

3. **Answer two questions:**
   - **Do the programs behave the same?** The change does affect program behavior, as the data can now include NaN values -> Choose 0 for NO 
   - **On a scale of 1 to 5, how realistic is this mistake?** A developer can forget to remove NaN values before processing data, so this looks natural -> Choose 4 or 5 for STRONGLY NATURAL

### Step 3: Use Additional Information When Uncertain
1. Below each mutant ID, you'll find a commit URL
2. Click the URL to view the original code context on GitHub
3. Use GitHub's "Search within code" to locate the relevant file
4. Use "View file" to see the complete file for better context
