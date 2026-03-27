<script setup lang="ts">
import type { PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import '@milaboratories/graph-maker/styles';
import { computed } from 'vue';
import { useApp } from '../app';

const app = useApp();

const defaultOptions = computed((): PredefinedGraphOption<'scatterplot'>[] | null => {
  const pCols = app.model.outputs.temporalLinePCols;
  if (!pCols || pCols.length === 0) return null;

  const freqCol = pCols.find((p) => p.spec.name === 'pl7.app/vdj/frequency');
  if (!freqCol) return null;

  // axes: [elementId, timepoint], value: frequency
  return [
    { inputName: 'x', selectedSource: freqCol.spec.axesSpec[1] },
    { inputName: 'y', selectedSource: freqCol.spec },
    { inputName: 'grouping', selectedSource: freqCol.spec.axesSpec[0] },
  ];
});
</script>

<template>
  <GraphMaker
    v-model="app.model.ui.temporalLineState"
    chartType="scatterplot"
    :p-frame="app.model.outputs.temporalLinePf"
    :defaultOptions="defaultOptions"
  />
</template>
