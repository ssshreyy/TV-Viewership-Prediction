import { Routes, RouterModule } from '@angular/router';
import { SentimentComponent } from './sentiment.component';

const childRoutes: Routes = [
    {
        path: '',
        component: SentimentComponent
    }
];

export const routing = RouterModule.forChild(childRoutes);
