<script setup lang="ts">
import type { PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import '@milaboratories/graph-maker/styles';
import { computed } from 'vue';
import { useApp } from '../app';

const app = useApp();

const defaultOptions = computed((): PredefinedGraphOption<'discrete'>[] | null => {
  const pCols = app.model.outputs.prevalenceHistogramPCols;
  if (!pCols || pCols.length === 0) return null;

  const countCol = pCols.find((p) => p.spec.name === 'pl7.app/vdj/cloneCount');
  if (!countCol) return null;

  // primaryGrouping = prevalenceCount (discrete axis), Y = cloneCount (value)
  return [
    { inputName: 'primaryGrouping', selectedSource: countCol.spec.axesSpec[0] },
    { inputName: 'y', selectedSource: countCol.spec },
  ];
});
</script>

<template>
  <GraphMaker
    v-model="app.model.ui.prevalenceHistogramState"
    chartType="discrete"
    :p-frame="app.model.outputs.prevalenceHistogramPf"
    :defaultOptions="defaultOptions"
  />
</template>
