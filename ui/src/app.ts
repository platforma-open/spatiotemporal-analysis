import { model } from '@platforma-open/milaboratories.spatiotemporal-analysis.model';
import { defineAppV3 } from '@platforma-sdk/ui-vue';
import { watch } from 'vue';
import MainPage from './pages/MainPage.vue';
import HeatmapPage from './pages/HeatmapPage.vue';
import TemporalLinePage from './pages/TemporalLinePage.vue';
import PrevalenceHistogramPage from './pages/PrevalenceHistogramPage.vue';

export const sdkPlugin = defineAppV3(model, (app) => {
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

// Ensure label fields are initialized after model state is loaded
const unwatch = watch(sdkPlugin, ({ loaded }) => {
  if (!loaded) return;
  const app = useApp();
  app.model.data.customBlockLabel ??= '';
  unwatch();
});
