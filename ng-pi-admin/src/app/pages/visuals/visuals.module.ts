import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { routing } from './visuals.routing';
import { SharedModule } from '../../shared/shared.module';
import { VisualsComponent } from './visuals.component';
import { HttpClientModule } from "@angular/common/http";
import { FormsModule } from '@angular/forms'
import { NgxEchartsModule } from 'ngx-echarts';

@NgModule({
    imports: [
        NgxEchartsModule,
        CommonModule,
        SharedModule,
        routing,
        HttpClientModule,
        FormsModule
    ],
    declarations: [
        VisualsComponent
    ]
})
export class VisualsModule { }
