<script setup lang="ts">
import type { PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import '@milaboratories/graph-maker/styles';
import { computed } from 'vue';
import { useApp } from '../app';

const app = useApp();

const defaultOptions = computed((): PredefinedGraphOption<'histogram'>[] | null => {
  const pCols = app.model.outputs.prevalenceHistogramPCols;
  if (!pCols || pCols.length === 0) return null;

  const countCol = pCols.find((p) => p.spec.name === 'pl7.app/vdj/cloneCount');
  if (!countCol) return null;

  // axes: [prevalenceCount], value: cloneCount
  return [
    { inputName: 'value', selectedSource: countCol.spec },
    { inputName: 'grouping', selectedSource: countCol.spec.axesSpec[0] },
  ];
});
</script>

<template>
  <GraphMaker
    v-model="app.model.ui.prevalenceHistogramState"
    chartType="histogram"
    :p-frame="app.model.outputs.prevalenceHistogramPf"
    :defaultOptions="defaultOptions"
  />
</template>
