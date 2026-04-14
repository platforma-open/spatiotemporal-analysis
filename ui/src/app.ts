import { model } from '@platforma-open/milaboratories.spatiotemporal-analysis.model';
import { defineAppV3 } from '@platforma-sdk/ui-vue';
import MainPage from './pages/MainPage.vue';
import HeatmapPage from './pages/HeatmapPage.vue';
import TemporalLinePage from './pages/TemporalLinePage.vue';
import PrevalenceHistogramPage from './pages/PrevalenceHistogramPage.vue';

export const sdkPlugin = defineAppV3(model, (app) => {
  app.model.data.customBlockLabel ??= '';

  return {
    progress: () => {
      return app.model.outputs.isRunning;
    },
    routes: {
      '/': () => MainPage,
      '/heatmap': () => HeatmapPage,
      '/temporal': () => TemporalLinePage,
      '/prevalence': () => PrevalenceHistogramPage,
    },
  };
});

export const useApp = sdkPlugin.useApp;
