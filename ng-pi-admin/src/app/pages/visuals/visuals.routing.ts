import { Routes, RouterModule } from '@angular/router';
import { VisualsComponent } from './visuals.component';

const childRoutes: Routes = [
    {
        path: '',
        component: VisualsComponent
    }
];

export const routing = RouterModule.forChild(childRoutes);
