# Liver Disease Workbench: UI Enhancement Specifications (Implementation Ready)

## 1. Medical Paradigm Selector - Global Header

### UI Placement: Global Header Dropdown (SELECTED)
Located in the main header, persistent across all tabs. Changes the entire patient view context.

### Paradigm Options

#### 1. **Liver Disease Progression** (Primary)
**Focus:** Early hepatic encephalopathy detection, decompensation risk
**Key Features:**
- Wearables: REM sleep %, sleep fragmentation, HRV, circadian rhythm
- EHR: Ammonia, MELD score, cognitive tests (NCT-A/B), medication compliance
**Models:** XGBoost, LSTM for temporal patterns
**Accent Color:** #F59E0B (Warm Orange)

#### 2. **Cirrhosis Management**
**Focus:** Preventing complications (varices, ascites, HCC)
**Key Features:**
- Wearables: Activity levels (fatigue), sleep quality, weight trends
- EHR: Liver enzymes, platelet count, albumin, INR, imaging results
**Accent Color:** #DC2626 (Deep Red)

#### 3. **Post-Transplant Monitoring**
**Focus:** Rejection detection, medication compliance
**Key Features:**
- Wearables: Activity, sleep quality, temperature (infection risk)
- EHR: Immunosuppressant levels, liver enzymes, biopsy results
**Accent Color:** #10B981 (Green)

#### 4. **Metabolic Syndrome**
**Focus:** MASLD/NASH, diabetes, cardiovascular risk
**Key Features:**
- Wearables: Activity, glucose (if CGM), HRV, sleep
- EHR: HbA1c, lipids, BMI, blood pressure, liver fat quantification
**Accent Color:** #8B5CF6 (Purple)

#### 5. **General Internal Medicine**
**Focus:** Holistic health monitoring
**Key Features:**
- Wearables: All available metrics
- EHR: Vitals, labs, medications
**Accent Color:** #3B82F6 (Blue)

### UI Design: Paradigm Selector Component

**Button (in header):**
- Background: Current paradigm accent color
- Text: White, 15px, weight 600
- Icon + Label + Dropdown arrow
- Padding: 12px 24px
- Border radius: 12px
- Shadow: 0 4px 6px rgba(0,0,0,0.1)

**Dropdown Menu:**
- Width: 350px
- White background
- Border radius: 12px
- Shadow: 0 10px 15px rgba(0,0,0,0.1)
- Border: 1px solid #E5E7EB

**Each Option:**
- Padding: 16px 20px
- Border-bottom: 1px solid #E5E7EB (except last)
- Hover: Background #F9FAFB
- Active: Background #EFF6FF, left border 3px accent color
- Layout: Icon (24px emoji) + Content + Checkmark (if active)

**Content Structure:**
- Paradigm Name: 14px, weight 600, #1A2332
- Description: 13px, #6B7280

---

## 2. Cohort Comparison - Toggle Chips/Pills

### Concept
Compare patient's biomarkers against multiple cohorts simultaneously using multi-select toggle chips.

### UI Placement Options

**Option A: Data Explorer Tab - Top Bar** (SELECTED)
- Always visible when viewing time series
- Horizontal row of chips above the main chart area
- Easy to toggle on/off different cohorts

**Option B: Floating Comparison Panel**
- Collapsible side panel that overlays the chart
- Can be pinned open or minimized
- More screen real estate for charts

**Option C: Chart Legend Integration**
- Chips integrated into the chart legend itself
- Clicking chip toggles that cohort's visibility
- Most compact option

### Toggle Chip UI Design

**Inactive Chip:**
- Background: #FFFFFF
- Border: 1.5px solid #E5E7EB
- Text: #6B7280, 14px, weight 500
- Padding: 8px 16px
- Border radius: 20px (full pill shape)
- Cursor: pointer
- Transition: all 0.2s ease

**Active Chip:**
- Background: #EFF6FF
- Border: 1.5px solid #6366F1
- Text: #6366F1, 14px, weight 600
- Icon: Small checkmark or dot before text
- Shadow: 0 2px 4px rgba(99,102,241,0.15)

**Hover (Inactive):**
- Background: #F9FAFB
- Border: 1.5px solid #D1D5DB

**Hover (Active):**
- Transform: translateY(-1px)
- Shadow: 0 4px 6px rgba(99,102,241,0.2)

### Available Cohorts

```
[Patient Only] [All Patients (140)] [Stage-Matched (47)] [Age-Matched (23)] [Custom...]
```

**Chip Labels:**
1. **Patient Only** - Default selected, shows just this patient's data
2. **All Patients (140)** - Entire database
3. **Stage-Matched: Child-Pugh B (47)** - Same disease stage
4. **Age-Matched: 60-70, Male (23)** - Demographic match
5. **Custom Selection...** - Opens modal to select specific patients

### Visual Representation on Charts

When cohorts are active:
- Show percentile bands (25th, 50th, 75th)
- Patient's line overlays on top
- Each cohort gets a subtle background color
- Legend shows: Patient (solid line) + Cohort names (shaded areas)

**Example:**
```
┌─────────────────────────────────────────────┐
│ Cohorts: [Patient Only] [✓ All Patients]   │
│          [✓ Stage-Matched] [Age-Matched]    │
├─────────────────────────────────────────────┤
│           REM Sleep % Over Time             │
│                                              │
│ 100% ┌───────────────────────────────────┐ │
│      │ ░░░░░ Stage cohort 75th %ile     │ │
│  75% │ ▓▓▓▓▓ All patients range         │ │
│  50% │ ════  Cohort medians             │ │
│  25% │                                   │ │
│      │  ●──●──●  PT-1847 (bold line)    │ │
│   0% └───────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 3. Data Source Selector

### Concept
Choose which data streams to include in analysis with per-source configuration.

### UI Design: Multi-Source Toggle

**Card Layout:**
- Grouped by category: WEARABLES | CLINICAL SYSTEMS | PATIENT INPUT
- Each group has uppercase label (11px, weight 600, letter-spacing 1px)

**Source Item:**
- Checkbox (20×20, rounded 4px)
  - Unchecked: Border 2px #E5E7EB
  - Checked: Background accent color, white checkmark
- Label + Status in vertical stack
  - Label: 14px, weight 500, #1A2332
  - Status: 12px, #6B7280 (e.g., "127 days", "Not connected")
- Action button (right-aligned)
  - "Configure" (primary) or "Connect" (secondary)
  - Padding: 6px 12px, 12px font, rounded 6px

**Source States:**
- Active (checked): Background #EFF6FF, border #6366F1
- Inactive: Background white, border #E5E7EB
- Hover: Background #F9FAFB

### Available Sources

**WEARABLES:**
- ✓ Oura Ring (127 days) - [Configure]
- ☐ Apple Watch (Not connected) - [Connect]
- ☐ Continuous Glucose Monitor - [Connect]

**CLINICAL SYSTEMS:**
- ✓ EHR - Epic (Last sync: 2h ago) - [Sync Now]
- ☐ Laboratory LIS - [Connect]
- ☐ PACS/Imaging - [Connect]

**PATIENT INPUT:**
- ☐ Symptom Diary - [Enable]
- ☐ Medication Tracking - [Enable]
- ☐ Quality of Life Surveys - [Enable]

---

## 4. Optimized Features for Liver Disease Progression

### Feature Set (Pre-configured when "Liver Disease Progression" paradigm selected)

#### WEARABLE BIOMARKERS (Oura Ring)

**Primary (Always Enabled):**
1. ✓ **REM Sleep %** ⭐ [Critical] - Most predictive for HE
2. ✓ **REM Fragmentation** ⭐ [High] - Count of REM interruptions
3. ✓ **Deep Sleep %** ⭐ - Restorative sleep quality
4. ✓ **Sleep Latency** ⭐ - Time to fall asleep (circadian disruption)
5. ✓ **HRV Balance** ⭐ [High] - Autonomic dysfunction indicator
6. ✓ **Nighttime HRV** ⭐ - Specifically 2am-6am window
7. ✓ **Body Temperature Deviation** ⭐ - Metabolic stress marker
8. ✓ **Resting Heart Rate** - Baseline cardiac function

**Secondary (Optionally Enabled):**
9. ☐ Sleep Efficiency %
10. ☐ Total Sleep Duration
11. ☐ Wake After Sleep Onset (WASO)
12. ☐ Respiratory Rate
13. ☐ Activity Score
14. ☐ Low Activity Alert

#### CLINICAL BIOMARKERS (EHR)

**Liver Function:**
1. ✓ **Ammonia (NH₃)** ⭐⭐ [Critical] - Direct HE predictor
2. ✓ **Albumin** ⭐ - Synthetic function
3. ✓ **Bilirubin** ⭐ - Total & direct
4. ✓ **INR** ⭐ - Coagulation
5. ✓ **ALT** - Hepatocellular injury
6. ✓ **AST** - Hepatocellular injury

**Cognitive & Neurological:**
7. ✓ **NCT-A Time** ⭐⭐ [Critical] - Number Connection Test A
8. ✓ **NCT-B Time** ⭐⭐ [Critical] - Number Connection Test B
9. ☐ **Hepatic Encephalopathy Grade** - When documented

**Disease Severity:**
10. ✓ **MELD Score** ⭐⭐ [Critical] - Calculated from Cr, Bili, INR
11. ✓ **Child-Pugh Score** ⭐ - Clinical staging

**Medications:**
12. ✓ **Lactulose Compliance** ⭐ - Doses/day
13. ✓ **Rifaximin Compliance** ⭐ - % doses taken
14. ☐ **Diuretic Use**
15. ☐ **Beta-blocker Use**

### Feature Grouping in UI

**Collapsible Categories:**
- SLEEP ARCHITECTURE ▼ (5/8 selected)
- PHYSIOLOGICAL ▼ (4/6 selected)
- LIVER FUNCTION ▼ (6/6 selected)
- COGNITIVE FUNCTION ▼ (2/3 selected)
- DISEASE SEVERITY ▼ (2/2 selected)
- MEDICATIONS ▼ (2/4 selected)

**Each Feature Item:**
- Checkbox (18×18)
- Feature name with importance stars
- Badge (if Critical or High)
- Current value (right-aligned, monospace font)

**Importance Indicators:**
- ⭐ = High importance (orange star #F59E0B)
- ⭐⭐ = Critical (double stars)

**Badges:**
- Critical: Background #FEE2E2, Text #991B1B
- High: Background #FEF3C7, Text #92400E

**Summary Panel (bottom of feature list):**
- Light blue gradient background
- Border: 2px solid accent color
- Shows: Total selected, Wearable count, Clinical count, Expected AUC range

---

## 5. Visual Design Refinements

### Color Coding by Paradigm
Each paradigm gets accent colors used in:
- Paradigm selector button background
- Active tab highlights
- Alert badges
- Chart accent colors
- Checkbox checked states
- Border accents

**Color Mapping:**
- Liver Disease: #F59E0B (Warm Orange)
- Cirrhosis: #DC2626 (Deep Red)
- Post-Transplant: #10B981 (Green)
- Metabolic: #8B5CF6 (Purple)
- General: #3B82F6 (Blue)

### Typography Hierarchy
- Paradigm name: 16px, weight 600
- Card titles: 18px, weight 600
- Feature category headers: 13px, uppercase, weight 600, letter-spacing 1px
- Feature names: 14px, weight 400
- Feature values: 12px, JetBrains Mono, weight 500
- Importance stars: 14px emoji
- Badges: 11px, weight 600
- Help text: 13px, color secondary

### Spacing Consistency
- Card padding: 24px
- Section gaps: 24px
- Item gaps: 12px
- Inline gaps: 8px
- Button padding: 12px 24px
- Chip padding: 8px 16px

### Interactive States
All interactive elements should have:
- Hover state (slight background change or lift)
- Active/selected state (accent color)
- Smooth transitions (0.2s ease)
- Proper cursor (pointer for clickable)

---

## Implementation Priority

**Phase 1 (Core Features):**
1. Medical Paradigm Selector in header
2. Data Source toggles in sidebar
3. Liver Disease feature set

**Phase 2 (Comparison):**
4. Cohort comparison toggle chips
5. Percentile band visualization on charts

**Phase 3 (Polish):**
6. Paradigm-specific color theming
7. Feature configuration persistence
8. Export/save configurations

This creates a comprehensive, paradigm-aware workbench optimized for liver disease monitoring while maintaining flexibility for other conditions!
