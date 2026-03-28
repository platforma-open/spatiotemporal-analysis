<script setup lang="ts">
import type { PObjectId, PlRef } from '@platforma-sdk/model';
import { getSingleColumnData } from '@platforma-sdk/model';
import {
  PlAccordion,
  PlAccordionSection,
  PlAgDataTableV2,
  PlBlockPage,
  PlBtnGhost,
  PlBtnGroup,
  PlDropdown,
  PlDropdownRef,
  PlElementList,
  PlMaskIcon24,
  PlNumberField,
  PlSlideModal,
  PlTooltip,
  usePlDataTableSettingsV2,
} from '@platforma-sdk/ui-vue';
import { computed, ref, watch, watchEffect } from 'vue';
import { useApp } from '../app';

const app = useApp();

const settingsAreShown = ref(app.model.outputs.datasetSpec === undefined);
const showSettings = () => {
  settingsAreShown.value = true;
};

function setInput(inputRef?: PlRef) {
  app.model.args.abundanceRef = inputRef;
}

const calculationModeOptions = [
  { label: 'Population-Level', value: 'population' },
  { label: 'Intra-Subject', value: 'intra-subject' },
];

const normalizationOptions = [
  { label: 'Relative Frequency', value: 'relative-frequency' },
  { label: 'CLR Transform', value: 'clr' },
];

const tableSettings = usePlDataTableSettingsV2({
  model: () => app.model.outputs.mainTable,
});

// Auto-build default subtitle from selected settings
watchEffect(() => {
  const metaOpts = app.model.outputs.metadataOptions ?? [];
  const findLabel = (ref?: string) => {
    if (!ref) return undefined;
    const opt = metaOpts.find((o: { value?: string; label?: string }) => o.value === ref);
    return opt?.label;
  };

  const parts: string[] = [];

  const mode = app.model.args.calculationMode;
  parts.push(mode === 'intra-subject' ? 'Intra-Subject' : 'Population');

  const groupLabel = findLabel(app.model.args.groupingColumnRef);
  if (groupLabel) parts.push(groupLabel);

  const temporalLabel = findLabel(app.model.args.temporalColumnRef);
  if (temporalLabel) parts.push(temporalLabel);

  app.model.args.defaultBlockLabel = parts.join(', ');
});

// Subject is required only in intra-subject mode
const subjectRequired = computed(() => app.model.args.calculationMode === 'intra-subject');

// Fetch unique timepoint values from the temporal column via pframe driver
const timepointValues = ref<string[]>([]);

watch(
  () => ({
    pframe: app.model.outputs.temporalColumnPframe,
    colId: app.model.outputs.temporalColumnId,
  }),
  async ({ pframe, colId }) => {
    if (!pframe || !colId) {
      timepointValues.value = [];
      return;
    }
    try {
      const colData = await getSingleColumnData(pframe, colId as PObjectId);
      const unique = [...new Set(
        colData.data
          .filter((v): v is string | number => v != null)
          .map(String),
      )].sort();
      timepointValues.value = unique;
    } catch {
      timepointValues.value = [];
    }
  },
  { immediate: true },
);

const availableTimepointsToAdd = computed(() => {
  const current = new Set(app.model.args.timepointOrder);
  return timepointValues.value.filter((v: string) => !current.has(v));
});

const resetTimepointOrder = () => {
  if (timepointValues.value.length) {
    app.model.args.timepointOrder = [...timepointValues.value];
  }
};

// Auto-populate timepoint order when temporal column changes
const temporalSyncCol = ref<string | undefined>(app.model.args.temporalColumnRef);
watchEffect(() => {
  const col = app.model.args.temporalColumnRef;
  const vals = timepointValues.value;

  if (vals && vals.length > 0) {
    const current = app.model.args.timepointOrder;
    const valSet = new Set(vals);
    const hasInvalidItems = current.some((v: string) => !valSet.has(v));

    if (col !== temporalSyncCol.value || current.length === 0) {
      app.model.args.timepointOrder = [...vals];
      temporalSyncCol.value = col;
    } else if (hasInvalidItems) {
      app.model.args.timepointOrder = current.filter((v: string) => valSet.has(v));
    }
  }
});

const isTimepointOrderOpen = ref(true);
const isAdvancedOpen = ref(false);
</script>

<template>
  <PlBlockPage
    v-model:subtitle="app.model.args.customBlockLabel"
    :subtitle-placeholder="app.model.args.defaultBlockLabel"
    title="Spatiotemporal Analysis"
  >
    <template #append>
      <PlBtnGhost @click.stop="showSettings">
        Settings
        <template #append>
          <PlMaskIcon24 name="settings" />
        </template>
      </PlBtnGhost>
    </template>

    <PlAgDataTableV2
      v-model="app.model.ui.tableState"
      :settings="tableSettings"
      not-ready-text="Data is not computed"
      show-export-button
      show-search-field
    />
  </PlBlockPage>

  <PlSlideModal v-model="settingsAreShown">
    <template #title>Settings</template>

    <PlDropdownRef
      v-model="app.model.args.abundanceRef"
      :options="app.model.outputs.abundanceOptions"
      label="Select abundance"
      clearable
      required
      @update:model-value="setInput"
    />

    <PlBtnGroup
      v-model="app.model.args.calculationMode"
      :options="calculationModeOptions"
      label="Calculation mode"
    />

    <PlDropdown
      v-model="app.model.args.subjectColumnRef"
      :options="app.model.outputs.metadataOptions"
      label="Subject variable"
      :required="subjectRequired"
      clearable
    >
      <template #tooltip>
        Required for Intra-Subject mode. Optional in Population-Level mode —
        when omitted, all samples are pooled and cross-subject metrics are skipped.
      </template>
    </PlDropdown>

    <PlDropdown
      v-model="app.model.args.groupingColumnRef"
      :options="app.model.outputs.metadataOptions"
      label="Grouping variable"
      clearable
    />

    <PlDropdown
      v-model="app.model.args.temporalColumnRef"
      :options="app.model.outputs.metadataOptions"
      label="Temporal variable"
      clearable
    />

    <PlAccordion multiple>
      <PlAccordionSection
        v-if="app.model.args.temporalColumnRef"
        v-model="isTimepointOrderOpen"
        label="Timepoint order"
      >
        <div style="display: flex; margin-bottom: -15px;">
          Define timepoint order
          <PlTooltip class="info">
            <template #label>Define timepoint order</template>
            <template #tooltip>
              <div>
                Drag to reorder timepoints chronologically.
                The order determines how temporal metrics
                (TSI, Log2 Peak Delta, Log2 Kinetic Delta)
                are computed. First = earliest, last = latest.
              </div>
            </template>
          </PlTooltip>
        </div>
        <PlElementList v-model:items="app.model.args.timepointOrder">
          <template #item-title="{ item }">
            {{ item }}
          </template>
        </PlElementList>
        <PlBtnGhost
          v-if="availableTimepointsToAdd.length > 0"
          @click="resetTimepointOrder"
        >
          Reset to default
          <template #append>
            <PlMaskIcon24 name="reverse" />
          </template>
        </PlBtnGhost>
      </PlAccordionSection>

      <PlAccordionSection v-model="isAdvancedOpen" label="Advanced settings">
        <PlBtnGroup
          v-model="app.model.args.normalization"
          :options="normalizationOptions"
          label="Normalization"
        />

        <PlNumberField
          v-model="app.model.args.minAbundanceThreshold"
          label="Min abundance threshold"
          :min-value="0"
          :step="1"
        >
          <template #tooltip>
            Filter clones with abundance below this value in ALL samples before computation.
            Default 0 includes everything.
          </template>
        </PlNumberField>

        <PlNumberField
          v-model="app.model.args.minSubjectCount"
          label="Min subject count"
          :min-value="1"
          :step="1"
        >
          <template #tooltip>
            Averaged cross-subject metrics (Mean RI, Mean Log2PD, etc.) are set to NaN
            when a clone is present in fewer subjects than this threshold. Default: 2.
          </template>
        </PlNumberField>

        <PlNumberField
          v-model="app.model.args.topN"
          label="Top N clones (temporal plot)"
          :min-value="1"
          :step="1"
        >
          <template #tooltip>
            Number of top clones to show in the temporal trajectory plot,
            ranked by absolute Log2 Peak Delta. Default: 20.
          </template>
        </PlNumberField>

        <PlNumberField
          v-model="app.model.args.presenceThreshold"
          label="Presence threshold"
          :min-value="0"
          :max-value="1"
          :step="0.0001"
        >
          <template #tooltip>
            Minimum frequency for a clone to be considered present in a group.
            Default 0 means any detection counts.
          </template>
        </PlNumberField>

        <PlNumberField
          v-model="app.model.args.pseudoCount"
          label="Pseudo-count"
          :min-value="0"
          :step="1"
        >
          <template #tooltip>
            Added to frequencies before log2 fold-change computation to prevent log(0).
            Default: 1.
          </template>
        </PlNumberField>
      </PlAccordionSection>
    </PlAccordion>
  </PlSlideModal>
</template>
