import { Routes, RouterModule } from '@angular/router';
import { PreprocessComponent } from './preprocess.component';

const childRoutes: Routes = [
    {
        path: '',
        component: PreprocessComponent
    }
];

export const routing = RouterModule.forChild(childRoutes);
