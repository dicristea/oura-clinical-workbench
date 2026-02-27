# Claude Code Implementation Guide: Liver Disease Workbench UI Enhancements

## Overview
This guide contains 5 sequential prompts to enhance your patient monitoring workbench with:
- Medical paradigm selector (view patients through different clinical lenses)
- Data source toggles (Oura, EHR, etc.)
- Cohort comparison chips (compare patient to populations)
- Liver disease-optimized feature selection
- Full integration with paradigm-aware theming

## Implementation Order

### ✅ Prompt 1: Medical Paradigm Selector
**File:** `prompt-1-paradigm-selector.md`
**Estimated Time:** 1-2 hours
**Description:** Adds a dropdown in the header to switch between 5 clinical paradigms (Liver Disease, Cirrhosis, Post-Transplant, Metabolic, General Medicine)
**Key Components:** ParadigmSelector button, dropdown menu with 5 options
**Dependencies:** None
**Deliverable:** Working paradigm selector with visual feedback

---

### ✅ Prompt 2: Data Sources Panel
**File:** `prompt-2-data-sources.md`
**Estimated Time:** 1-2 hours
**Description:** Adds a sidebar card for toggling data sources (Oura Ring, EHR, labs, etc.) grouped by category
**Key Components:** DataSourcesPanel with checkboxes and action buttons
**Dependencies:** None (independent of Prompt 1)
**Deliverable:** Working data source toggles with proper styling

---

### ✅ Prompt 3: Cohort Comparison Chips
**File:** `prompt-3-cohort-comparison.md`
**Estimated Time:** 1-2 hours
**Description:** Adds toggle chips above Data Explorer charts to compare patient against cohorts (All Patients, Stage-Matched, etc.)
**Key Components:** CohortComparison with multi-select toggle chips
**Dependencies:** None (independent of previous prompts)
**Deliverable:** Working cohort chips with multi-select logic

---

### ✅ Prompt 4: Feature Selection UI
**File:** `prompt-4-feature-selection.md`
**Estimated Time:** 2-3 hours
**Description:** Restructures feature selection into collapsible categories with importance indicators and liver disease-specific features
**Key Components:** FeatureSelection with 6 categories, 30 features, summary panel
**Dependencies:** None (can work standalone)
**Deliverable:** Complete feature selection UI with all categories and features

---

### ✅ Prompt 5: Integration & Paradigm-Aware Features
**File:** `prompt-5-integration.md`
**Estimated Time:** 2-3 hours
**Description:** Wires everything together - paradigm changes update features and accent colors throughout the UI
**Key Components:** ParadigmContext, paradigm configuration, color theming
**Dependencies:** Requires Prompts 1 and 4 to be completed
**Deliverable:** Fully integrated system where paradigm selection updates the entire interface

---

## Total Estimated Time
**8-12 hours** of focused development work

## Recommended Approach

### Option A: Sequential Implementation (Recommended)
Complete prompts in order 1 → 2 → 3 → 4 → 5
- **Pros:** Each prompt builds on stable foundation, easier to debug
- **Cons:** Takes longer to see full functionality

### Option B: Parallel Development
Complete prompts 1, 2, 3, 4 independently, then do prompt 5
- **Pros:** Faster to get basic UI in place
- **Cons:** Integration (prompt 5) may require adjustments

### Option C: Minimum Viable Product
Complete prompts 1, 4, and 5 first (core paradigm + features)
Add prompts 2 and 3 later (data sources and cohort comparison)
- **Pros:** Fastest path to paradigm-aware feature selection
- **Cons:** Missing some nice-to-have features

## Architecture Notes

### State Management
The prompts suggest using React Context for sharing paradigm state:
```
<ParadigmProvider>
  <App>
    <Header>
      <ParadigmSelector /> {/* Prompt 1 */}
    </Header>
    <Sidebar>
      <DataSourcesPanel /> {/* Prompt 2 */}
      <FeatureSelection /> {/* Prompt 4 */}
    </Sidebar>
    <MainContent>
      <DataExplorer>
        <CohortComparison /> {/* Prompt 3 */}
        <Charts />
      </DataExplorer>
      <ModelLab />
    </MainContent>
  </App>
</ParadigmProvider>
```

### Color Theming
Use CSS custom properties for dynamic paradigm colors:
```css
:root {
  --accent-color: #F59E0B; /* Default: Liver Disease orange */
}

.checkbox.checked {
  background: var(--accent-color);
}
```

Update via JavaScript when paradigm changes:
```typescript
document.documentElement.style.setProperty('--accent-color', newColor);
```

### Data Flow
```
User selects paradigm
  ↓
ParadigmContext updates
  ↓
├─→ FeatureSelection updates checked features
├─→ CSS variables update colors
└─→ Toast shows confirmation
```

## Testing Checklist

After completing all prompts, verify:
- [ ] All 5 paradigms are selectable
- [ ] Paradigm selector shows correct icon, name, color
- [ ] Selecting paradigm updates feature checkboxes
- [ ] Selecting paradigm updates UI accent colors
- [ ] Data source checkboxes work
- [ ] Cohort chips toggle properly
- [ ] Multiple cohorts can be selected
- [ ] Feature categories collapse/expand
- [ ] Feature selection counts are accurate
- [ ] Summary panel shows correct totals
- [ ] Reset button works for current paradigm
- [ ] Selected paradigm persists on page refresh
- [ ] All transitions are smooth
- [ ] No console errors

## Future Enhancements (Not in these prompts)

After completing the 5 prompts, you could add:
- Actual cohort data visualization on charts (percentile bands)
- Data source configuration modals
- Custom cohort selection modal
- Paradigm-specific alerts/recommendations
- Save/load feature configurations
- Export paradigm settings
- Keyboard shortcuts for paradigm switching
- Paradigm comparison view (side-by-side)

## Support

Each prompt includes:
- ✅ Clear requirements
- ✅ Visual specifications with exact measurements
- ✅ Acceptance criteria
- ✅ Example code structures
- ✅ Notes on edge cases

If you get stuck, refer to:
- The working HTML prototype: `enhanced-workbench.html`
- The comprehensive spec: `workbench-spec-implementation.md`
- Visual mockups from earlier in this conversation

## Questions?

Common issues and solutions:
- **Q: Checkboxes not updating when paradigm changes**
  - A: Make sure FeatureSelection component is listening to ParadigmContext
- **Q: Colors not updating throughout UI**
  - A: Ensure all components use CSS variables or theme context
- **Q: Paradigm not persisting on refresh**
  - A: Check localStorage implementation in ParadigmSelector

Good luck with implementation! 🚀
