import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { routing } from './sentiment.routing';
import { SharedModule } from '../../shared/shared.module';
import { SentimentComponent } from './sentiment.component';
import { HttpClientModule } from "@angular/common/http";
import { FormsModule } from '@angular/forms';
import { NgxPaginationModule } from 'ngx-pagination';

@NgModule({
    imports: [
        NgxPaginationModule,
        CommonModule,
        SharedModule,
        routing,
        HttpClientModule,
        FormsModule
    ],
    declarations: [
        SentimentComponent
    ]
})
export class SentimentModule { }
