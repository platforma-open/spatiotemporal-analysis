<script setup lang="ts">
import type { PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import { computed } from 'vue';
import { useApp } from '../app';

const app = useApp();

const defaultOptions = computed((): PredefinedGraphOption<'heatmap'>[] | null => {
  const pCols = app.model.outputs.heatmapPCols;
  if (!pCols || pCols.length === 0) return null;

  const freqCol = pCols.find((p) => p.spec.name === 'pl7.app/vdj/normalizedFrequency');
  if (!freqCol) return null;

  // axes: [elementId, groupCategory], value: normalizedFrequency
  return [
    { inputName: 'x', selectedSource: freqCol.spec.axesSpec[1] },
    { inputName: 'y', selectedSource: freqCol.spec.axesSpec[0] },
    { inputName: 'value', selectedSource: freqCol.spec },
  ];
});
</script>

<template>
  <GraphMaker
    v-model="app.model.data.heatmapState"
    chartType="heatmap"
    :p-frame="app.model.outputs.heatmapPf"
    :defaultOptions="defaultOptions"
  />
</template>
