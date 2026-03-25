<script setup lang="ts">
import type { PlRef } from '@platforma-sdk/model';
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
import { computed, ref, watchEffect } from 'vue';
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

// Unique timepoint values from the model (extracted from p-column spec)
const timepointValues = computed(() => app.model.outputs.temporalColumnValues ?? []);

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
    title="In Vivo Compartment Analysis"
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
      required
    />

    <PlDropdown
      v-model="app.model.args.compartmentColumnRef"
      :options="app.model.outputs.metadataOptions"
      label="Compartment variable"
    />

    <PlDropdown
      v-model="app.model.args.temporalColumnRef"
      :options="app.model.outputs.metadataOptions"
      label="Temporal variable"
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
                (Temporal Shift Index, Log2 Kinetic Delta)
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
          v-model="app.model.args.presenceThreshold"
          label="Presence threshold"
          :min-value="0"
          :max-value="1"
          :step="0.0001"
        >
          <template #tooltip>
            Minimum frequency for a clone to be considered present in a compartment.
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
            Added to frequencies before computing Log2 Kinetic Delta to prevent log(0).
            Default: 1.
          </template>
        </PlNumberField>
      </PlAccordionSection>
    </PlAccordion>
  </PlSlideModal>
</template>
