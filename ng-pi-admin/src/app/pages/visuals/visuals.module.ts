import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { routing } from './visuals.routing';
import { SharedModule } from '../../shared/shared.module';
import { VisualsComponent } from './visuals.component';

@NgModule({
    imports: [
        CommonModule,
        SharedModule,
        routing
    ],
    declarations: [
        VisualsComponent
    ]
})
export class VisualsModule { }
